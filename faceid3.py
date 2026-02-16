import cv2
import face_recognition

# Запуск веб-камеры
video_capture = cv2.VideoCapture(0)

print("Запуск системы распознавания лиц...")

# Загружаем изображения известных лиц
known_face_encodings = []
known_face_names = ["Zhannur", "Abdurakhim", "Timur", "Erkebulan", "Zhuldyz", "Dinara", "Darkhan", "Iskander", "Damir"]  # Подпиши имена
known_face_images = ["known_faces/zhannur.jpg", "known_faces/abdurakhim.jpg", "known_faces/timur.jpg", "known_faces/erkebulan.jpg", "known_faces/zhuldyz.jpg", "known_faces/dinara.jpg", "known_faces/darkhan.jpg", "known_faces/iskander.jpg", "known_faces/damir.jpg"]

for i, image_path in enumerate(known_face_images):
    image = face_recognition.load_image_file(image_path)
    encodings = face_recognition.face_encodings(image)
    if encodings:
        known_face_encodings.append(encodings[0])
        print(f"✓ Лицо {known_face_names[i]} загружено!")
    else:
        print(f"⚠️ Лицо {known_face_names[i]} не найдено в {image_path}!")

print("Все лица загружены. Начинаем работу камеры...")

while True:
    # Захват кадра с веб-камеры
    ret, frame = video_capture.read()
    if not ret:
        print("Ошибка при захвате кадра с камеры!")
        break
    
    # Преобразование BGR -> RGB (face_recognition использует RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Определение местоположения лиц
    face_locations = face_recognition.face_locations(rgb_frame)
    face_landmarks = face_recognition.face_landmarks(rgb_frame)  # Добавили landmarks
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
        name = "Unknown"

        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        # Отображение рамки и имени
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Вывод видео
    cv2.imshow('Face Recognition', frame)

    # Выход из цикла по нажатию "q"
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождаем ресурсы
video_capture.release()
cv2.destroyAllWindows()
print("Выход из программы.")
