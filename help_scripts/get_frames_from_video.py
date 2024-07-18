import cv2
import os
from tkinter import Tk, filedialog


def select_file(initial_dir):
    root = Tk()
    root.withdraw()  # Закрыть главное окно
    file_selected = filedialog.askopenfilename(
        initialdir=initial_dir,
        filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")])
    return file_selected


def select_folder(initial_dir):
    root = Tk()
    root.withdraw()  # Закрыть главное окно
    folder_selected = filedialog.askdirectory(initialdir=initial_dir)
    return folder_selected


def play_video(video_path, save_folder):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    cv2.namedWindow('Video', cv2.WINDOW_NORMAL)  # Создание масштабируемого окна

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_pos = 0

    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame.")
            break

        cv2.imshow('Video', frame)

        key = cv2.waitKey(0)

        if key == 27:  # ESC key to exit
            break
        elif key == 81 or key == 2424832:  # Left arrow key
            frame_pos = max(0, frame_pos - 1)
        elif key == 83 or key == 2555904:  # Right arrow key
            frame_pos = min(frame_count - 1, frame_pos + 1)
        elif key == 84 or key == 2621440:  # Down arrow key
            save_path = os.path.join(save_folder, f'frame_{frame_pos}.png')
            cv2.imwrite(save_path, frame)
            print(f'Saved frame {frame_pos} as {save_path}')

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    initial_dir = "~/Видео/"  # Укажите путь к вашей начальной директории
    video_path = select_file(initial_dir)  # Открываем диалоговое окно для выбора видеофайла
    if video_path:
        save_folder = select_folder(initial_dir)  # Открываем диалоговое окно для выбора папки
        if save_folder:
            play_video(video_path, save_folder)
        else:
            print("No folder selected.")
    else:
        print("No video selected.")
