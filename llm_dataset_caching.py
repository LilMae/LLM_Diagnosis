import argparse
import os
from data.dataset import get_dataset
from data.llm_dataset import get_llm_dataset
from tqdm import tqdm
import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Cornstarch 멀티모달 + TRL GRPO 통합 예제")
    
    parser.add_argument("--cache_dir", type=str, default='./llm_cache', help="llm_model 캐싱할 폴더")
    parser.add_argument("--llm_name", type=str, default='Qwen/Qwen3-4B-Instruct-2507', help="LLM 모델 이름")
    
    # 데이터셋 옵션
    parser.add_argument("--batch_size",    type=int,   default=1,     help="미니배치 크기")
    parser.add_argument("--data_root",    type=str,   default='',     help="dataset 경로")
    parser.add_argument("--train_domain",    type=str, nargs='+',   default=['vat', 'mfd', 'dxai', 'vbl'],     help="dataset 경로")
    parser.add_argument("--valid_domain",    type=str, nargs='+',   default=['vat', 'mfd', 'dxai', 'vbl'],     help="dataset 경로")
    
    # LLM 데이터셋 옵션
    parser.add_argument("--embedding_model",    type=str,   default='sentence-transformers/paraphrase-multilingual-mpnet-base-v2',     help="RAG를 구축할 임베딩 모델")
    parser.add_argument("--docs_path",    type=str,   default='/workspace/docs_path',     help="RAG에 사용될 문서 폴더")
    parser.add_argument("--retriever_k",    type=int,   default=4,     help="RAG에서 Retreive 개수")

    args = parser.parse_args()

    """
    Dataloader 생성
    변환 순서 : data.dataset.VibrationDataset -> data.llm_dataset.LLM_Dataset -> torch.utils.data.DataLoader
    """
    train_dataset, val_dataset = get_dataset(args, train_domain=args.train_domain,
                valid_domain=args.valid_domain)
    train_llm_dataset, val_llm_dataset = get_llm_dataset(args, train_dataset, val_dataset)
    
    
    exp_name = 'LLM_Train_' + '+'.join(args.train_domain) + '_Val_' + '+'.join(args.valid_domain)
    data_cache_root = os.path.join(args.cache_dir, exp_name)
    os.makedirs(data_cache_root, exist_ok=False)
    
    train_pt = os.path.join(data_cache_root, 'train.pt')
    valid_pt = os.path.join(data_cache_root, 'valid.pt')
    
    # 데이터셋 캐싱
    all_valid_data = []
    print('ValidSet Testing')
    for data_sample in tqdm(val_llm_dataset, dynamic_ncols=True):
        all_valid_data.append(data_sample)
    torch.save(all_valid_data, valid_pt)
    
    
    all_train_data = []
    print('TrainSet Testing')
    for data_sample in tqdm(train_llm_dataset, dynamic_ncols=True):
        all_train_data.append(data_sample)
    torch.save(all_train_data, train_pt)
    
    
    
    