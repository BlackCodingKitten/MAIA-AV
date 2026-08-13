import os
import ffmpeg

# 1. Definiamo i percorsi delle cartelle
input_dir = "data/input/video"
output_dir = "data/input/mute"

# 2. Crea la cartella di output se non esiste già (evita errori)
os.makedirs(output_dir, exist_ok=True)

# 3. Ciclo per scorrere i numeri da 1 a 100
for i in range(1, 101):
    # Crea il nome del file aggiungendo gli zeri (es: video001.mp4, video015.mp4...)
    nome_file = f"video{i:03d}.mp4"
    
    percorso_input = os.path.join(input_dir, nome_file)
    percorso_output = os.path.join(output_dir, nome_file)
    
    # 4. Verifica che il video originale esista prima di procedere
    if os.path.exists(percorso_input):
        try:
            # Esegue l'operazione in una riga (copia il video, toglie l'audio)
            ffmpeg.input(percorso_input).output(percorso_output, an=None, vcodec='copy').run(overwrite_output=True, quiet=True)
            print(f"✓ Muto salvato: {nome_file}")
        except Exception as e:
            print(f"✗ Errore con {nome_file}: {e}")
    else:
        print(f"⚠ File non trovato, salto: {nome_file}")

print("\nOperazione completata su tutti i video!")