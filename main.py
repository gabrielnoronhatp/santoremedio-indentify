import cv2
from ultralytics import YOLO
from datetime import datetime
import tkinter as tk
from threading import Thread
from sort.sort import Sort
import numpy as np
import time


LOG_ARQUIVO = "log_detectados.txt"

def salvar_log(mensagem):
    
    with open(LOG_ARQUIVO, "a") as log:
        log.write(f"{datetime.now()} - {mensagem}\n")


class ContadorDePessoasApp:
    def __init__(self, root):
      
        self.root = root
        self.root.title("Santo Remedio - Identify")
        self.root.geometry("300x150")
        self.root.configure(bg="#4bb9ba")
        
        self.contador_cumulativo = 0  
        self.ids_rastreados = set()  
        self.tracker = Sort()  

        self.num_pessoas = tk.IntVar(value=0)

        self.label_contador = tk.Label(root, text="Pessoas detectadas:", font=("Arial", 16))
        self.label_contador.pack(pady=10)

        self.label_valor = tk.Label(root, textvariable=self.num_pessoas, font=("Arial", 24), fg="blue")
        self.label_valor.pack()

        self.btn_iniciar = tk.Button(root, text="Iniciar Detecção", command=self.iniciar_detectar)
        self.btn_iniciar.pack(pady=10)

        self.btn_parar = tk.Button(root, text="Parar Detecção", command=self.parar_detectar)
        self.btn_parar.pack(pady=10)

        self.rodando = False

    def iniciar_detectar(self):
      
        if not self.rodando:
            self.rodando = True
            Thread(target=self.detectar_pessoas, daemon=True).start()

    def detectar_pessoas(self):
   
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
            detecoes = []  

            
            for box in resultados[0].boxes:
                classe = int(box.cls[0])
                if modelo.names[classe] == "person":
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    score = float(box.conf[0])
                    detecoes.append([x1, y1, x2, y2, score])

            rastreamentos = self.tracker.update(np.array(detecoes))

            for rastreamento in rastreamentos:
                x1, y1, x2, y2, rastreador_id = map(int, rastreamento)
                if rastreador_id not in self.ids_rastreados:
                    self.ids_rastreados.add(rastreador_id)
                    self.contador_cumulativo += 1
                    salvar_log(f"Nova pessoa detectada. Total: {self.contador_cumulativo}")

              
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID: {rastreador_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

           
            self.num_pessoas.set(self.contador_cumulativo)
            cv2.imshow("Detecção de Pessoas", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        captura.release()
        cv2.destroyAllWindows()
        self.rodando = False

    def parar_detectar(self):
     
        self.rodando = False


if __name__ == "__main__":
    root = tk.Tk()
    app = ContadorDePessoasApp(root)
    root.mainloop()
