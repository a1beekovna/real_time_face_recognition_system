import cv2
import face_recognition
import time
import tkinter as tk
from tkinter import ttk

video_capture = cv2.VideoCapture(0)

print("Запуск системы распознавания лиц...")

known_face_encodings = []
known_face_names = ["Zhannur", "Abdurakhim", "Timur", "Zhuldyz", "Darkhan", "Iskander", "Damir"]
known_face_images = ["known_faces/zhannur.jpg", "known_faces/abdurakhim.jpg", "known_faces/timur.jpg", 
                      "known_faces/zhuldyz.jpg", "known_faces/darkhan.jpg", "known_faces/iskander.jpg", "known_faces/damir.jpg"]

face_counts = {name: 0 for name in known_face_names}
last_seen = {name: 0 for name in known_face_names}
unique_faces = set()
total_faces_detected = 0

for i, image_path in enumerate(known_face_images):
    image = face_recognition.load_image_file(image_path)
    encodings = face_recognition.face_encodings(image)
    if encodings:
        known_face_encodings.append(encodings[0])
        print(f"✓ Лицо {known_face_names[i]} загружено!")
    else:
        print(f"⚠️ Лицо {known_face_names[i]} не найдено в {image_path}!")

print("Все лица загружены. Начинаем работу камеры...")

root = tk.Tk()
root.title("📊 Статистика лиц")
root.geometry("500x350")

frame = ttk.Frame(root)
frame.pack(pady=10)

tree = ttk.Treeview(frame, columns=("Name", "Count"), show="headings", height=len(known_face_names))
tree.column("Name", width=150)
tree.column("Count", width=100)
tree.heading("Name", text="Имя")
tree.heading("Count", text="Количество")
tree.pack()

stats = {}
for name in known_face_names:
    stats[name] = tree.insert("", "end", values=(name, face_counts[name]))

# Метрики
metrics_frame = ttk.LabelFrame(root, text="📊 Метрики")
metrics_frame.pack(pady=10, fill="both", expand=True)

unique_faces_label = ttk.Label(metrics_frame, text="Уникальные лица: 0")
unique_faces_label.pack(pady=5)

recognition_rate_label = ttk.Label(metrics_frame, text="Процент распознавания: 0%")
recognition_rate_label.pack(pady=5)

# Функция обновления таблицы и метрик
def update_table():
    global total_faces_detected
    if total_faces_detected > 0:
        recognition_rate = (len(unique_faces) / total_faces_detected) * 100
    else:
        recognition_rate = 0
    
    for name in known_face_names:
        tree.item(stats[name], values=(name, face_counts[name]))
    
    unique_faces_label.config(text=f"Уникальные лица: {len(unique_faces)}")
    recognition_rate_label.config(text=f"Процент распознавания: {recognition_rate:.2f}%")
    
    root.after(1000, update_table)

# Кнопка закрытия
def close_app():
    video_capture.release()
    cv2.destroyAllWindows()
    root.destroy()

btn_close = tk.Button(root, text="Закрыть", command=close_app, font=("Arial", 12), bg="red", fg="white")
btn_close.pack(pady=10)

update_table()

# Основной цикл обработки видео
while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Ошибка при захвате кадра с камеры!")
        break
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    
    total_faces_detected += len(face_encodings)
    current_time = time.time()
    
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
        name = "Unknown"
        
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]
            unique_faces.add(name)
            
            if current_time - last_seen[name] > 2:
                face_counts[name] += 1
                last_seen[name] = current_time
        
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    cv2.imshow('Face Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
root.mainloop()
