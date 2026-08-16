import argparse, os, torch, torchaudio, ChatTTS

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default="你好呀，我是莉莉丝")
    args = ap.parse_args()

    os.makedirs("output", exist_ok=True)

    chat = ChatTTS.Chat()
    chat.load(compile=False)  # CPU 环境用 compile=False

    torch.manual_seed(42)  # 固定音色（不传 seed 给 sample_random_speaker）
    spk = chat.sample_random_speaker()
    params = ChatTTS.Chat.InferCodeParams(spk_sem=spk, temperature=0.3)

    wavs = chat.infer([args.text], params_infer_code=params)
    wav = wavs[0]
    try:
        torchaudio.save("output/lilith.wav", torch.from_numpy(wav).unsqueeze(0), 24000)
    except Exception:
        torchaudio.save("output/lilith.wav", torch.from_numpy(wav), 24000)
    print("saved output/lilith.wav len:", len(wav))

    os.system("ffmpeg -y -i output/lilith.wav -codec:a libmp3lame -qscale:a 4 output/lilith.mp3")
    print("saved output/lilith.mp3")

if __name__ == "__main__":
    main()
