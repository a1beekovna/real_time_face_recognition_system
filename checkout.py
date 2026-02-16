import cv2
import face_recognition

# Запуск веб-камеры
video_capture = cv2.VideoCapture(0)

while True:
    # Захват кадра с веб-камеры
    ret, frame = video_capture.read()
    
    # Преобразование BGR -> RGB (face_recognition использует RGB)
    rgb_frame = frame[:, :, ::-1]

    # Определение местоположения лиц
    face_locations = face_recognition.face_locations(rgb_frame)

    # Отображение рамок вокруг лиц
    for top, right, bottom, left in face_locations:
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

    # Вывод видео
    cv2.imshow('Face Recognition', frame)

    # Выход из цикла по нажатию "q"
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождаем ресурсы
video_capture.release()
cv2.destroyAllWindows()

