import cv2
from ultralytics import YOLO
from datetime import datetime
import tkinter as tk
from threading import Thread


LOG_ARQUIVO = "log_detectados.txt"

# Função para salvar logs
def salvar_log(mensagem):
    with open(LOG_ARQUIVO, "a") as log:
        log.write(f"{datetime.now()} - {mensagem}\n")


class ContadorDePessoasApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Contador de Pessoas")
        self.root.geometry("300x150")
        self.contador_estavel = 0 
        self.num_pessoas_temp = [] 

      
        self.num_pessoas = tk.IntVar(value=0)

        self.label_contador = tk.Label(root, text="Pessoas detectadas:", font=("Arial", 16))
        self.label_contador.pack(pady=10)

        self.label_valor = tk.Label(root, textvariable=self.num_pessoas, font=("Arial", 24), fg="blue")
        self.label_valor.pack()

        self.btn_iniciar = tk.Button(root, text="Iniciar Detecção", command=self.iniciar_detectar)
        self.btn_iniciar.pack(pady=10)

        self.rodando = False

    def iniciar_detectar(self):
        if not self.rodando:
            self.rodando = True
            self.thread = Thread(target=self.detectar_pessoas)
            self.thread.start()

    def atualizar_contador_estavel(self):
     
        if len(self.num_pessoas_temp) >= 10: 
            moda = max(set(self.num_pessoas_temp), key=self.num_pessoas_temp.count)
            if moda != self.contador_estavel:
                self.contador_estavel = moda
                self.num_pessoas.set(self.contador_estavel)
                salvar_log(f"Atualizado: {self.contador_estavel} pessoas")
            self.num_pessoas_temp = [] 

    def detectar_pessoas(self):
        # Carregar o modelo YOLO
        modelo = YOLO("yolov8n.pt")
        captura = cv2.VideoCapture(0)

        if not captura.isOpened():
            print("Erro ao acessar a câmera.")
            self.rodando = False
            return

        while self.rodando:
            ret, frame = captura.read()
            if not ret:
                print("Erro ao ler o frame da câmera.")
                break

            resultados = modelo(frame)
            num_pessoas_frame = 0

            # Processar detecções
            for box in resultados[0].boxes:
                classe = int(box.cls[0])  
                if modelo.names[classe] == "person":  
                    num_pessoas_frame += 1
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = box.conf[0]
                    label = f"{modelo.names[classe]} {conf:.2f}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

         
            self.num_pessoas_temp.append(num_pessoas_frame)
            self.atualizar_contador_estavel()

        
            cv2.imshow("Detecção de Pessoas", frame)

           
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        captura.release()
        cv2.destroyAllWindows()
        self.rodando = False


if __name__ == "__main__":
    root = tk.Tk()
    app = ContadorDePessoasApp(root)
    root.mainloop()
