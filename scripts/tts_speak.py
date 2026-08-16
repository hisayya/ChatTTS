import argparse, os, wave, struct, torch, ChatTTS

def save_wav_std(data, path, sr=24000):
    """用标准库写 16-bit PCM wav（避免 torchaudio/torchcodec 依赖）"""
    data = (data * 32767).astype('int16')
    with wave.open(path, 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(data.tobytes())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default="你好呀，我是莉莉丝")
    args = ap.parse_args()

    os.makedirs("output", exist_ok=True)

    chat = ChatTTS.Chat()
    chat.load(compile=False)  # CPU 环境用 compile=False

    torch.manual_seed(42)
    spk = chat.sample_random_speaker()
    params = ChatTTS.Chat.InferCodeParams(spk_emb=spk, temperature=0.3)

    wavs = chat.infer([args.text], params_infer_code=params)
    wav = wavs[0]
    print("synthesized len:", len(wav))

    save_wav_std(wav, "output/lilith.wav")
    print("saved output/lilith.wav")

    os.system("ffmpeg -y -i output/lilith.wav -codec:a libmp3lame -qscale:a 4 output/lilith.mp3")
    print("saved output/lilith.mp3")

if __name__ == "__main__":
    main()
