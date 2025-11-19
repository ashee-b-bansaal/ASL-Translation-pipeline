from utils import say_hello, read_from_folder, parse_nonmanuals, vis, round_robin_split, DataSplitter, CNNDataset, ctc_loss_weighted
from models import ResNetWindowCTC, ResNetWindowLSTMCTC, TemporalResNetCTC, Imu1dImage2dCTC, greedy_decode, wer, greedy_decode_blank, ctc_loss_weighted
from ctc_normalized import Imu1dImage2dCTCNormalized

import os
import logging
import argparse
import numpy as np
from collections import Counter
import random
from collections import defaultdict

import torch
import torch.nn as nn
import torchvision.models as models
from torch import Tensor

import os, torch, torch.nn as nn
from torch.cuda.amp import autocast, GradScaler

print(say_hello("multi-face"))

parser = argparse.ArgumentParser(description="Train ResNet-CTC model with k-fold cross-validation")
parser.add_argument("--output", type=str, default="feature_outputs", help="Folder name to save raw outputs")
parser.add_argument("--path", type=str, default="./", help="Folder name for dataset")
parser.add_argument("--test", type=str, default=None, help="Folder name for testset")
parser.add_argument("--exclude", type=str, default=None, help="Folder name for exclusion")
parser.add_argument("--gpu", type=str, default=0, help="GPU")
parser.add_argument("--mode", type=str, default='emotion', help="Mode")
parser.add_argument("--include", type=str, default=None, help="Folder name for inclusion")
parser.add_argument("--ch", type=str, default='0,1,2,3', help="channel -set 0,1,2,3")
parser.add_argument("--poi", type=str, default='300,350', help="point of interest")
parser.add_argument("--target_sessions", type=str, default=None, help="target sessions")
parser.add_argument("--windows", type=str, default='160,80', help="windows")
parser.add_argument("--fold", type=int, default=0, help="Which fold to run (1-k). Use 0 to run all folds.")
parser.add_argument("--k_folds", type=int, default=10, help="Total number of folds for cross-validation")
parser.add_argument("--normalize_k", type=int, default=None, help="If set, normalize logits time length to K using adaptive pooling (inference and training)")
parser.add_argument("--target_timeline_length", type=int, default=20, help="Target timeline length for consistent output")

args = parser.parse_args()
feature_folder = args.output
folder_path = args.path
test_sessions = args.test
exclude_sessions = args.exclude
gpu_num = args.gpu
mode = args.mode
include_sessions = args.include
ch_num = args.ch
poi = args.poi
target_sessions = args.target_sessions
windows = args.windows
fold_to_run = args.fold
k_folds = args.k_folds

TARGET_NUM_THREADS = '4'
os.environ['OMP_NUM_THREADS'] = TARGET_NUM_THREADS
os.environ['OPENBLAS_NUM_THREADS'] = TARGET_NUM_THREADS
os.environ['MKL_NUM_THREADS'] = TARGET_NUM_THREADS
os.environ['VECLIB_MAXIMUM_THREADS'] = TARGET_NUM_THREADS
os.environ['NUMEXPR_NUM_THREADS'] = TARGET_NUM_THREADS

# Base path for results - will be updated for each fold
base_fn_path = "results/%s_%s_%s_%s_%s_%s"%(feature_folder, 
                                                    str(test_sessions),
                                                      str(mode), 
                                                      "ch"+str(len(ch_num.split(','))),
                                                     "poi"+str(poi.split(',')[0])+str(poi.split(',')[1]),  
                                                     "w"+str(windows.split(',')[0])+str(windows.split(',')[1]))

def ensure_folder_exists(folder_path):
    """Check if a folder exists, and create it if not."""
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"✅ Folder created: {folder_path}")
    else:
        print(f"📂 Folder already exists: {folder_path}")

def get_logger(log_file):
    logger = logging.getLogger(log_file)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(log_file, mode='w')
        fh.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
        logger.addHandler(fh)
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
        logger.addHandler(sh)
    return logger

def print_and_log(content, log_file="main_logfile.txt"):
    logger = get_logger(log_file)
    print(content)
    logger.info(content)

# Load data
dp = folder_path
data_path = folder_path + '/session_'
tmp_dp = sorted(os.listdir(dp))
data_sessions = [i.split('_')[1] for i in tmp_dp if i.find('session') == 0]

if target_sessions !=None:
    data_sessions =target_sessions.split(',')
else:
    data_sessions = [i.split('_')[1] for i in tmp_dp if i.find('session') == 0]
print_and_log(data_sessions)

if exclude_sessions != None:
    print_and_log(f"Excluded session: {exclude_sessions}")
    for i in exclude_sessions.split(','):
        data_sessions.remove(i)

if include_sessions !=None:
    data_sessions = include_sessions.split(',')

all_data = []
all_data_truth = []

print(data_sessions)

# Mode configuration
if mode == 'signs':
    poi_list = '340,600'.split(',')
    target_height = int((int(poi_list[1]) - int(poi_list[0]))*0.95)
    poc = [3]
    batch_size = 30
    window_size, window_step =160,80

elif mode == 'emotion':
    poi_list = '300,400'.split(',')
    target_height = int((int(poi_list[1]) - int(poi_list[0]))*0.95)
    poc = [0,1,2,3]
    batch_size = 50
    window_size, window_step =40,20

elif mode =='grammar':
    poi_list = '300,325'.split(',')
    target_height = int((int(poi_list[1]) - int(poi_list[0]))*0.95)
    poc = [3]
    batch_size = 10
    window_size, window_step =80,40

elif mode == 'mouth':
    poi_list = '300,430'.split(',')
    target_height = int((int(poi_list[1]) - int(poi_list[0]))*0.95)
    poc = [0,1,2,3]
    batch_size = 10
    window_size, window_step =80,40

elif mode == 'emomouth':
    poi_list = '330,380'.split(',')
    target_height = int((int(poi_list[1]) - int(poi_list[0]))*0.95)
    poc = [3]
    batch_size = 10
    window_size, window_step =100,40

elif mode == 'all':
    poi_list = [int(i) for i in poi.split(',')]
    target_height = int((int(poi_list[1]) - int(poi_list[0]))*0.95)
    poc = [int(i) for i in ch_num.split(',')]
    batch_size = 30
    window_size, window_step = 140, 35

elif mode == 'isolated':
    poi_list = [int(i) for i in poi.split(',')]
    target_height = int((int(poi_list[1]) - int(poi_list[0]))*0.95)
    poc = [int(i) for i in ch_num.split(',')]
    print(poc)
    batch_size = 30
    window_size, window_step =100,40

print_and_log(f"{mode, poi_list, target_height, poc, batch_size, window_size, window_step}")

print_and_log('Loading all data...')
# Set global random seed to ensure consistent data splitting across different modes
np.random.seed(42)
random.seed(42)
print_and_log("🎯 Set global random seed=42 for consistent data splits across different modes")

for i, p in enumerate(data_sessions):
    print_and_log(' Loading from %s' % p)
    this_all_data, this_all_data_truth = read_from_folder(
        p,
        data_path,
        is_train=True,
        is_shuffle = True,
        seed_num = data_sessions[i],
        use_timestamps=False,
        prefer_legacy_dirs=True
    )
    # Store session info and sample index with each data item
    this_all_data_with_meta = []
    for idx, data_item in enumerate(this_all_data):
        # Add session number and original sample index to the tuple
        this_all_data_with_meta.append((*data_item, p, idx))
    all_data += this_all_data_with_meta
    all_data_truth += this_all_data_truth

# Extract labels
lable = []
for i in all_data_truth:
    kkk = parse_nonmanuals(i[3], mode)
    print(kkk)
    for j in kkk.split(','):
        if j != '':
            lable.append(j)
        else:
            print(">>>>>>")

# Define vocabulary
facial_markers = {"none", "raise", "furrow", "shake", "mm", "th", "puff", "cs", "oo", "cha", "pah"}
emotional_markers = {"happy", "sad", "angry", "scared", "surprised", "disgust"}
mouth_morphemes = { "oo", "mm", "cha", "cs", "th", "puff", "pah"}
grammatical_markers = {"raise", "furrow", "shake"}
emo_mouth = { "mm", "th", "puff", "cs", "oo", "cha", "pah","happy", "sad", "angry", "scared", "surprised", "disgust"}
all = { "mm", "th", "puff", "cs", "oo", "cha", "pah","happy", "sad", "angry", "scared", "surprised", "disgust","raise", "furrow", "shake"}

gnd_list = sorted(list(set(lable)))

if sorted(gnd_list) == sorted(list(grammatical_markers)):
    gnd_list = ["raise", "furrow", "shake"]
if sorted(gnd_list) == sorted(list(mouth_morphemes)):
    gnd_list = ["oo", "mm", "cha", "cs", "th", "puff", "pah"]
if sorted(gnd_list) == sorted(list(emotional_markers)):
    gnd_list = ["happy", "sad", "angry", "scared", "surprised", "disgust", 'others']
if sorted(gnd_list) == sorted(list(emo_mouth)):
    gnd_list = ["mm", "th", "puff", "cs", "oo", "cha", "pah","happy", "sad", "angry", "scared", "surprised", "disgust"]
if sorted(gnd_list) == sorted(list(all)):
    gnd_list = [ "mm", "th", "puff", "cs", "oo", "cha", "pah","happy", "sad", "angry", "scared", "surprised", "disgust","raise", "furrow", "shake"]

label_dic =  {value: index for index, value in enumerate(gnd_list)}
label_dic_reverse = {index: value for index, value in enumerate(gnd_list)}
class_num = len(gnd_list)
print_and_log(f"Number of Classes: {class_num}")
print_and_log(label_dic)
print_and_log(label_dic_reverse)

# K-fold cross-validation setup
data = all_data
k = k_folds
splits = round_robin_split(data, k)

# Determine which folds to run
if fold_to_run == 0:
    folds_to_run = list(range(k))
    print_and_log(f"🔄 Running ALL {k} folds for cross-validation")
else:
    if fold_to_run < 1 or fold_to_run > k:
        raise ValueError(f"Fold must be between 1 and {k}, or 0 for all folds. Got: {fold_to_run}")
    folds_to_run = [fold_to_run - 1]
    print_and_log(f"🎯 Running fold {fold_to_run} of {k}")

# Store results for all folds
all_fold_results = []

for fold_idx in folds_to_run:
    print_and_log(f"\n{'='*60}")
    print_and_log(f"🚀 STARTING FOLD {fold_idx + 1} of {k}")
    print_and_log(f"{'='*60}")
    
    # Create fold-specific folder
    fn_path = f"{base_fn_path}_k{k}_fold_{fold_idx + 1}"
    ensure_folder_exists(fn_path)
    
    # Set up fold-specific logging
    def print_and_log_fold(content, log_file=None):
        if log_file is None:
            log_file = fn_path + "/training_log.txt"
        logger = get_logger(log_file)
        print(content)
        logger.info(content)
    
    train = [x for j, f in enumerate(splits) if j != fold_idx for x in f]
    test = splits[fold_idx]
    print_and_log_fold(f"Fold {fold_idx + 1}: Train={len(train)}, Test={len(test)}", fn_path + "/training_log.txt")
    # Process training data
    train_data = []
    test_data = []
    
    for i in train:
        thd = parse_nonmanuals(i[2], mode)
        try:
            if thd != '':
                # Extract session and sample index (last two elements of i)
                session_num = i[-2]
                sample_idx = i[-1]
                train_data += [(i[0][poc,:,int(poi_list[0]):int(poi_list[1])], i[1], parse_nonmanuals(i[2], mode), i[2], session_num, sample_idx)]
        except:
            print_and_log_fold(f"Test: label error - {i[2]}")

    for i in test:
        thd = parse_nonmanuals(i[2], mode)
        print(thd)
        try:
            if thd != '':
                # Extract session and sample index (last two elements of i)
                session_num = i[-2]
                sample_idx = i[-1]
                test_data += [(i[0][poc,:,int(poi_list[0]):int(poi_list[1])], i[1], parse_nonmanuals(i[2], mode), i[2], session_num, sample_idx)]
        except:
            print_and_log_fold(f"Test: label error - {i[2]}")

    print_and_log_fold(f"training set: {len(train_data)}, testing set: {len(test_data)}")

    # Calculate class weights
    cnt = []
    for i in train:
        try:
            for j in parse_nonmanuals(i[2], mode).split(','):
                if j != '':
                    cnt.append(j)
        except:
            print_and_log_fold(f"error label: {i[2]}")
            pass

    freqs = Counter(cnt)
    max_freq = max(freqs.values())
    class_rarity = {lab: max_freq / f for lab, f in freqs.items()}
    print_and_log_fold(f"{freqs, class_rarity}")

    class_weights = torch.tensor([class_rarity[i] for i in gnd_list])

    # Create data loaders
    worker_num = min(8, os.cpu_count() or 4)
    data_splitter = DataSplitter(train_data, test_data, BATCH_SIZE=batch_size, WORKER_NUM=worker_num, target_height=target_height)
    train_loader = data_splitter.train_loader
    test_loader = data_splitter.test_loader

    # CUDA setup - removed expandable_segments as it's not available in older PyTorch versions
    # os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128")
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    device = int(gpu_num)
    scaler = GradScaler(enabled=True)

    # Model setup
    # Set target timeline length to ensure consistent output lengths
    # Use a fixed target length that works for both signs and expressions
    target_timeline_length = args.target_timeline_length
    
    if args.normalize_k is not None:
        model = Imu1dImage2dCTCNormalized(
            num_classes=class_num,
            C_img=len(poc),
            C_imu=3,
            window_size=window_size,
            window_step=window_step,
            normalize_k=int(args.normalize_k)
        ).to(device)
    else:
        model = Imu1dImage2dCTC(
            num_classes=class_num, 
            C_img=len(poc), 
            C_imu=3, 
            window_size=window_size, 
            window_step=window_step
        ).to(device)
    model = model.to(memory_format=torch.channels_last)

    def to_cuda(x):
        return x.to(device, non_blocking=True).to(memory_format=torch.channels_last)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    ctc_loss_fn = nn.CTCLoss(blank=class_num, zero_infinity=True)
    print_and_log_fold(f"Model: {model}")

    # Training loop
    num_epochs = 500
    best_test_loss = 100.0
    exact_match_accuracy = 0
    exact_match = 0.0
    avg_wer = 0.0
    avg_test_loss = 0.0  # Initialize avg_test_loss
    
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0

        for i, (input_arr_raw, targets_tuple) in enumerate(train_loader):
            # Handle metadata in training loop
            if len(targets_tuple) > 2:
                targets = (targets_tuple[0], targets_tuple[1])
                # Ignore metadata during training
            else:
                targets = targets_tuple
            
            input_arr = input_arr_raw[0]
            input_imu = input_arr_raw[1]
            x = to_cuda(Tensor(input_arr))
            x1 = to_cuda(Tensor(input_imu))
            x1 = x1.squeeze(1) 
            optimizer.zero_grad(set_to_none=True)
            with autocast():   
                ctc_logits = model(x, x1) 
                logp = ctc_logits.log_softmax(2).float()

                target_sequences = []
                for s in targets[0]:
                    tokens = s.split(',')
                    indices = [label_dic[token] for token in tokens]
                    target_sequences.append(torch.tensor(indices, dtype=torch.long).to(device))

                per_ex = ctc_loss_weighted(ctc_logits, target_sequences, device, class_num)
                weights = torch.stack([
                    torch.tensor((np.array([class_rarity[i] for i in t.split(",")])).sum()/len(t.split(',')))
                    for t in targets[0]
                ]).to(device)
                loss = (per_ex * weights).mean()

                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update()
           
                total_loss += loss.item()

        print_and_log_fold(f"Epoch {epoch+1}/{num_epochs} | Avg Loss: {total_loss / len(train_loader):.4f}")

        if epoch % 3 == 0:
            model.eval()
            all_outputs = []
            all_outputs_gnd = []
            all_outputs_pre = []
            test_total_loss = 0.0
            
            # Variables to store probability matrices (only saved if this becomes best model)
            prob_matrices_data = []
            
            with torch.no_grad():
                print_and_log_fold(f"--- Test set evaluation at Epoch {epoch} ---")
                
                for j, (test_arr_raw, test_targets_tuple) in enumerate(test_loader):
                    # Handle metadata extraction
                    if len(test_targets_tuple) > 2:
                        test_targets = (test_targets_tuple[0], test_targets_tuple[1])
                        test_metadata = test_targets_tuple[2]  # List of (session, sample) tuples
                    else:
                        test_targets = test_targets_tuple
                        test_metadata = None
                    
                    x = Tensor(test_arr_raw[0]).to(device)
                    x1 = Tensor(test_arr_raw[1]).to(device)
                    x1 = x1.squeeze(1) 
                    logits_test = model(x, x1)
                    log_probs_test = logits_test.log_softmax(2)
                    
                    # Store probability matrices data (only save if this becomes best model)
                    probs_test = torch.exp(log_probs_test)
                    
                    # Store raw logits and probabilities for each sample in the batch
                    for b in range(logits_test.size(1)):
                        sample_id = j * batch_size + b
                        
                        sample_logits = logits_test[:, b, :].cpu().numpy()
                        sample_probs = probs_test[:, b, :].cpu().numpy()
                        
                        # Get session and sample info if available
                        if test_metadata is not None:
                            session_num, sample_idx = test_metadata[b]
                        else:
                            session_num, sample_idx = None, None
                        
                        sample_metadata = {
                            'ground_truth': test_targets[0][b],
                            'vocabulary': gnd_list,
                            'blank_index': class_num,
                            'time_windows': sample_logits.shape[0],
                            'num_classes': class_num + 1,
                            'epoch': epoch,
                            'sample_id': sample_id,
                            'session': session_num,
                            'sample_index': sample_idx
                        }
                        
                        prob_matrices_data.append({
                            'sample_id': sample_id,
                            'logits': sample_logits,
                            'probabilities': sample_probs,
                            'metadata': sample_metadata
                        })

                    # Decode
                    predicted_strings, predicted_strings_raw = greedy_decode_blank(log_probs_test, label_dic_reverse, blank_index=class_num)
                    T_pred_test = logits_test.size(0)
                    input_lengths_test = torch.full(size=(len(test_targets[0]),), fill_value=T_pred_test, dtype=torch.long).to(device)

                    target_sequences_test = []
                    for s in test_targets[0]:
                        tokens = s.split(',')
                        indices = [label_dic[token] for token in tokens]
                        target_sequences_test.append(torch.tensor(indices, dtype=torch.long).to(device))

                    target_sequences_test_cp = target_sequences_test.copy()
                    targets_concat_test = torch.cat(target_sequences_test)
                    target_lengths_test = torch.tensor([len(t) for t in target_sequences_test], dtype=torch.long)
                    targets_concat_test = targets_concat_test.to(device)
                    target_lengths_test = target_lengths_test.to(device)

                    log_probs_test = log_probs_test.log_softmax(2)
                    test_loss_per = ctc_loss_fn(log_probs_test, targets_concat_test, input_lengths_test, target_lengths_test)

                    weights = torch.stack([
                        torch.tensor((np.array([class_rarity[i] for i in t.split(",")])).sum()/len(t.split(',')))
                        for t in test_targets[0]
                    ]).to(device)
                    test_loss = (test_loss_per * weights).mean()

                    test_total_loss += test_loss.item()

                    for b in range(logits_test.size(1)):
                        gt_str = test_targets[0][b]
                        pred_str = predicted_strings[b]
                        pred_str = pred_str.replace(' ', ',')
                        
                        # Add session and sample info if available
                        if test_metadata is not None:
                            session_num, sample_idx = test_metadata[b]
                            all_outputs.append(f"{gt_str} - {pred_str} <<<<< {predicted_strings_raw[b].replace(' ', ',')} >>>> {test_targets[1][b]} ----- {session_num}, {sample_idx}\n")
                        else:
                            all_outputs.append(f"{gt_str} - {pred_str} <<<<< {predicted_strings_raw[b].replace(' ', ',')} >>>> {test_targets[1][b]}\n")
                        
                        all_outputs_gnd.append(gt_str)
                        all_outputs_pre.append(pred_str)
                        if gt_str == pred_str:
                            exact_match = exact_match + 1

            avg_test_loss = test_total_loss / len(test_loader)

            # Save best checkpoint
            if avg_test_loss < best_test_loss and avg_test_loss > 0:
                best_test_loss = avg_test_loss
                
                # Save probability matrices for the best model
                prob_matrices_dir = f"{fn_path}/best_model_probability_matrices"
                
                import shutil
                if os.path.exists(prob_matrices_dir):
                    shutil.rmtree(prob_matrices_dir)
                
                os.makedirs(prob_matrices_dir, exist_ok=True)
                print_and_log_fold(f"💾 Saving probability matrices for BEST MODEL to: {prob_matrices_dir}")
                
                # Create a summary file explaining the saved data
                summary_file = f"{prob_matrices_dir}/README_probability_matrices.txt"
                with open(summary_file, "w") as f:
                    f.write("PROBABILITY MATRICES ANALYSIS - BEST MODEL\n")
                    f.write("=" * 50 + "\n\n")
                    f.write("This directory contains probability matrices for the BEST MODEL (lowest test loss).\n\n")
                    f.write("File formats:\n")
                    f.write("- sample_XXXX_logits.npy: Raw logits from model output (shape: Tw x V+1)\n")
                    f.write("- sample_XXXX_probabilities.npy: Probability distributions (shape: Tw x V+1)\n")
                    f.write("- sample_XXXX_metadata.npy: Metadata including ground truth and vocabulary\n\n")
                    f.write("Where:\n")
                    f.write("- Tw = number of time windows\n")
                    f.write("- V+1 = vocabulary size + 1 (includes blank token)\n")
                    f.write("- The last column (index V) is the blank token probability\n\n")
                    f.write("How to load and analyze:\n")
                    f.write("```python\n")
                    f.write("import numpy as np\n")
                    f.write("# Load probability matrix\n")
                    f.write("probs = np.load('sample_0000_probabilities.npy')\n")
                    f.write("metadata = np.load('sample_0000_metadata.npy', allow_pickle=True).item()\n")
                    f.write("vocabulary = metadata['vocabulary']\n")
                    f.write("ground_truth = metadata['ground_truth']\n")
                    f.write("blank_index = metadata['blank_index']\n\n")
                    f.write("# Find highest probability token at each time step\n")
                    f.write("predicted_indices = np.argmax(probs, axis=1)\n")
                    f.write("# Convert indices to words (excluding blank)\n")
                    f.write("predicted_words = [vocabulary[i] if i < len(vocabulary) else 'BLANK' for i in predicted_indices]\n")
                    f.write("```\n\n")
                    f.write(f"Best Epoch: {epoch}\n")
                    f.write(f"Best Test Loss: {avg_test_loss:.4f}\n")
                    f.write(f"Vocabulary: {gnd_list}\n")
                    f.write(f"Blank index: {class_num}\n")
                    f.write(f"Number of classes: {class_num + 1}\n")
                
                # Save all probability matrices for the best model
                total_saved_samples = 0
                for sample_data in prob_matrices_data:
                    sample_id = sample_data['sample_id']
                    
                    # Save logits
                    logits_file = f"{prob_matrices_dir}/sample_{sample_id:04d}_logits.npy"
                    np.save(logits_file, sample_data['logits'])
                    
                    # Save probabilities
                    probs_file = f"{prob_matrices_dir}/sample_{sample_id:04d}_probabilities.npy"
                    np.save(probs_file, sample_data['probabilities'])
                    
                    # Save metadata
                    metadata_file = f"{prob_matrices_dir}/sample_{sample_id:04d}_metadata.npy"
                    np.save(metadata_file, sample_data['metadata'], allow_pickle=True)
                    
                    total_saved_samples += 1
                
                print_and_log_fold(f"✅ Saved probability matrices for {total_saved_samples} test samples")
                
                with open(fn_path + "/best_outputs.txt", "w") as f:
                    f.writelines(all_outputs)
                print_and_log_fold("✅ Saved new best model and outputs!")
                
                exact_matches = 0
                gt_list = all_outputs_gnd
                pred_list = all_outputs_pre
                for gt, pred in zip(gt_list, pred_list):
                    if gt.strip() == pred.strip().replace(' ', ','):
                        exact_matches += 1

                total = len(gt_list)
                exact_match_accuracy = exact_matches / total

                wer_list = []
                for gt, pred in zip(gt_list, pred_list):
                    gt_tokens = gt.strip().split()[0]
                    pred_tokens = pred.strip().replace(' ', ',')
                    wer_score = wer(gt_tokens, pred_tokens)
                    wer_list.append(wer_score)

                avg_wer = np.mean(wer_list)

        print_and_log_fold(f"Epoch [{epoch+1}/{num_epochs}],Avg Test Loss: {avg_test_loss:.4f}, Best Test Loss: {best_test_loss:.4f}, Exact Match: {exact_match_accuracy:.2f}, Average WER: {avg_wer:.2%}, INFO: {mode, poi_list, target_height, poc, batch_size, window_size, window_step}")

    # Store results for this fold
    fold_result = {
        'fold': fold_idx + 1,
        'best_test_loss': best_test_loss,
        'exact_match_accuracy': exact_match_accuracy,
        'avg_wer': avg_wer,
        'best_epoch': epoch
    }
    all_fold_results.append(fold_result)
    
    print_and_log_fold(f"\n✅ COMPLETED FOLD {fold_idx + 1}")
    print_and_log_fold(f"   Best Test Loss: {best_test_loss:.4f}")
    print_and_log_fold(f"   Exact Match Accuracy: {exact_match_accuracy:.2%}")
    print_and_log_fold(f"   Average WER: {avg_wer:.2%}")
    print_and_log_fold(f"{'='*60}")

# Print summary of all folds
if len(all_fold_results) > 1:
    print_and_log(f"\n🎯 CROSS-VALIDATION SUMMARY ({len(all_fold_results)} folds)")
    print_and_log(f"{'='*80}")
    
    test_losses = [r['best_test_loss'] for r in all_fold_results]
    accuracies = [r['exact_match_accuracy'] for r in all_fold_results]
    wers = [r['avg_wer'] for r in all_fold_results]
    
    print_and_log(f"Test Loss: {np.mean(test_losses):.4f} ± {np.std(test_losses):.4f}")
    print_and_log(f"Accuracy:  {np.mean(accuracies):.2%} ± {np.std(accuracies):.2%}")
    print_and_log(f"WER:       {np.mean(wers):.2%} ± {np.std(wers):.2%}")
    
    print_and_log(f"\nIndividual Fold Results:")
    for result in all_fold_results:
        print_and_log(f"  Fold {result['fold']:2d}: Loss={result['best_test_loss']:.4f}, Acc={result['exact_match_accuracy']:.2%}, WER={result['avg_wer']:.2%}")
    
    print_and_log(f"{'='*80}")
