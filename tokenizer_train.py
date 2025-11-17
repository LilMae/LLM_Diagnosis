from transformers import AutoTokenizer, AutoModelForCausalLM
from tokenizer_trainer.models.ViT_pytorch import VisionTransformerAE

from data.dataset import OrderInvariantSignalImager, VibrationDataset

from tokenizer_trainer.vib_tokenizer import VibrationTokenizer, VibTokeizerTrainer

import lightning as L
import os
import argparse
import torch
from peft import get_peft_model, LoraConfig, TaskType
from torch.utils.data import DataLoader
from lightning.pytorch.loggers import WandbLogger

os.environ["TOKENIZERS_PARALLELISM"] = "false"

if __name__ =='__main__':
    parser = argparse.ArgumentParser(description='Vibration LLM training/evaluation script')
    # 데이터셋 관련 옵션들
    parser.add_argument('--data_root',   type=str, default='./data/processed', help='llm_dataset_caching.py를 통해 만들어진 데이터 pt파일경로')
    
    # 캐싱 경로 옵션들
    parser.add_argument('--model_cache',    type=str, default='./llm_cache', help='LLM 모델들을 caching해둘 경로 (TRANSFORMERS_CACHE)')
    
    # 학습 결과물 저장 옵션들
    parser.add_argument('--model_out',    type=str, default='./output', help='학습 결과가 저장될 디렉토리')
    parser.add_argument('--log_dir',    type=str, default='./log_output', help='학습 결과가 저장될 디렉토리')
    
    # LLM 모델 관련 옵션들
    parser.add_argument('--run_name',    type=str, default='1104', help='wandb에 저장될 run 이름')
    parser.add_argument('--llm_model',      type=str, default='Qwen/Qwen3-4B-Instruct-2507', help='LLM Model name')
    
    
    # 학습관련 옵션들
    parser.add_argument('--pretrained_path', type=str, default='./best.pth', help='pretrained vib_AE')
    parser.add_argument('--batch_size',    type=int, default=32, help='학습 배치사이즈')
    parser.add_argument('--max_epochs',    type=int, default=200, help='학습 epoch')
    args = parser.parse_args()
    
    
    # 1. LLM Model Setting
    tokenizer = AutoTokenizer.from_pretrained(args.llm_model,
                                                cache_dir=args.model_cache)
    llm = AutoModelForCausalLM.from_pretrained(args.llm_model, device_map="auto",
                                            cache_dir=args.model_cache)
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    llm = get_peft_model(llm, peft_config)
    llm.print_trainable_parameters()
    special_tokens = {
        'additional_special_tokens': ["<NORMAL_VIB_EMB>", "<CURRENT_VIB_EMB>"]
    }
    tokenizer.add_special_tokens(special_tokens)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
    tokenizer.padding_side = "right"
    llm.config.pad_token_id = tokenizer.pad_token_id
    llm.resize_token_embeddings(len(tokenizer))

    # 2. Vibration Tokenizer 세팅
    vib_ae = VisionTransformerAE(num_classes=5,)
    payload = torch.load(args.pretrained_path)
    state_dict = payload.get('model_state_dict', payload)
    vib_ae.load_state_dict(state_dict)
    
    vib_tokenizer = VibrationTokenizer(
                                        vib_ae=vib_ae,
                                        token_embed_dim=int(llm.get_input_embeddings().embedding_dim),
                                        freeze_encoder=True,
                                    )
    
    # 3. Dataset 세팅
    signal_imger = OrderInvariantSignalImager(
                                mode='stft+cross',
                                log1p=True,
                                normalize= "per_channel",  
                                eps=1e-8,
                                out_dtype=torch.float32,
                                max_order=20.0,           
                                H_out=224,                
                                W_out=224,               
                                stft_nperseg=1024,
                                stft_hop=256,
                                stft_window="hann",
                                stft_center=True,
                                stft_power=1.0,           
                            )
    vib_trainset = VibrationDataset(
                                data_root=args.data_root,
                                using_dataset = ['vat', 'vbl', 'mfd'],
                                window_sec=5,
                                stride_sec=3,
                                transform=signal_imger,
                                test_mode=False
                            )
    vib_valset = VibrationDataset(
                                data_root=args.data_root,
                                using_dataset = ['dxai'],
                                window_sec=5,
                                stride_sec=3,
                                transform=signal_imger,
                                test_mode=False
                            )
    
    train_loader = DataLoader(vib_trainset, batch_size=args.batch_size, shuffle=True,
                          num_workers=os.cpu_count()//2, pin_memory=True)
    val_loader = DataLoader(vib_valset, batch_size=args.batch_size, shuffle=False,
                            num_workers=os.cpu_count()//2, pin_memory=True)
    
    vib_tokenizer_lightning = VibTokeizerTrainer(
        vib_tokenizer=vib_tokenizer,
        llm=llm,
        tokenizer=tokenizer
    )
    wandb_logger = WandbLogger(
        project="vibration-tokenizer",   # 프로젝트 이름 (WandB 대시보드에서 확인)
        name=args.run_name,              # 실험 이름 (지금은 0904로 들어감)
        save_dir=args.log_dir            # 로그 저장 경로
    )

    vib_trainer = L.Trainer(
        max_epochs=args.max_epochs,
        logger=wandb_logger,
        log_every_n_steps=1
    )
    vib_trainer.fit(
        model = vib_tokenizer_lightning,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader
    )
