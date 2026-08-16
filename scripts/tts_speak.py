import argparse, os, numpy as np, torch, torchaudio, ChatTTS

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default="你好呀，我是莉莉丝")
    args = ap.parse_args()

    os.makedirs("output", exist_ok=True)

    chat = ChatTTS.Chat()
    chat.load(compile=False)  # CPU 环境用 compile=False

    # 固定一个温柔女声 speaker 向量（用固定种子，保证音色稳定）
    spk = chat.sample_random_speaker(seed=42)
    params = ChatTTS.Chat.InferCodeParams(spk_sem=spk, temperature=0.3)

    wavs = chat.infer([args.text], params_infer_code=params)
    wav = wavs[0]
    try:
        torchaudio.save("output/lilith.wav", torch.from_numpy(wav).unsqueeze(0), 24000)
    except Exception:
        torchaudio.save("output/lilith.wav", torch.from_numpy(wav), 24000)
    print("saved output/lilith.wav len:", len(wav))

    # 转 mp3
    os.system("ffmpeg -y -i output/lilith.wav -codec:a libmp3lame -qscale:a 4 output/lilith.mp3")
    print("saved output/lilith.mp3")

if __name__ == "__main__":
    main()
