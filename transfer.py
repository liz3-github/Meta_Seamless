import torch
import torchaudio
from transformers import AutoProcessor, SeamlessM4Tv2Model

def transcribe_audio(file_path, target_lang):
    # 加载模型和处理器
    processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
    model = SeamlessM4Tv2Model.from_pretrained("facebook/seamless-m4t-v2-large")

    # 加载音频文件
    audio, orig_freq = torchaudio.load(file_path)
    
    # 明确指定目标采样率为16000
    target_freq = 16000
    audio = torchaudio.functional.resample(audio, orig_freq=orig_freq, new_freq=target_freq)

    # 确保音频是单声道的
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)

    # 准备输入，明确指定采样率
    audio_inputs = processor(audios=audio.squeeze().numpy(), sampling_rate=target_freq, return_tensors="pt")

    # 运行ASR
    output = model.generate(**audio_inputs, tgt_lang=target_lang)

    # 解码输出
    transcription = processor.decode(output[0].tolist(), skip_special_tokens=True)
    
    return transcription

def save_transcription(transcription, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(transcription)

def main():
    # 音频文件路径
    audio_file_path = "v058_MECOLAB_16k.wav"  # 请替换为您的音频文件路径

    # 使用中文（普通话）作为目标语言进行转录
    print("开始中文转录...")
    chinese_transcription = transcribe_audio(audio_file_path, "cmn")
    print("中文转录完成。")
    save_transcription(chinese_transcription, "chinese_transcription.txt")
    print("中文转录已保存至 chinese_transcription.txt")

    # 使用英语作为目标语言进行转录
    print("\n开始英文转录...")
    english_transcription = transcribe_audio(audio_file_path, "eng")
    print("英文转录完成。")
    save_transcription(english_transcription, "english_transcription.txt")
    print("英文转录已保存至 english_transcription.txt")

    print("\n转录过程已完成。请查看输出文件。")

if __name__ == "__main__":
    main()