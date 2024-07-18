import cv2
import os
from tkinter import Tk, filedialog


def select_files(initial_dir):
    root = Tk()
    root.withdraw()  # Закрыть главное окно
    files_selected = filedialog.askopenfilenames(
        initialdir=initial_dir,
        filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")])
    return files_selected


def select_folder(initial_dir):
    root = Tk()
    root.withdraw()  # Закрыть главное окно
    folder_selected = filedialog.askdirectory(initialdir=initial_dir)
    return folder_selected


def play_videos(video_paths, save_folder):
    caps = [cv2.VideoCapture(video_path) for video_path in video_paths]

    if not all(cap.isOpened() for cap in caps):
        print("Error: Could not open all videos.")
        return

    cv2.namedWindow('Videos', cv2.WINDOW_NORMAL)  # Создание масштабируемого окна

    frame_counts = [int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) for cap in caps]
    frame_pos = 0

    while True:
        frames = []
        for cap in caps:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                break
            frames.append(frame)

        if not frames:
            break

        combined_frame = cv2.hconcat(frames)  # Объединяем кадры горизонтально
        cv2.imshow('Videos', combined_frame)

        key = cv2.waitKey(0)

        if key == 27:  # ESC key to exit
            break
        elif key == 81 or key == 2424832:  # Left arrow key
            frame_pos = max(0, frame_pos - 1)
        elif key == 83 or key == 2555904:  # Right arrow key
            frame_pos = min(min(frame_counts) - 1, frame_pos + 1)
        elif key == 84 or key == 2621440:  # Down arrow key
            for i, frame in enumerate(frames):
                save_path = os.path.join(save_folder, f'video_{i + 1}_frame_{frame_pos}.png')
                cv2.imwrite(save_path, frame)
                print(f'Saved frame {frame_pos} of video {i + 1} as {save_path}')

    for cap in caps:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    initial_dir = "~/Видео/"  # Укажите путь к вашей начальной директории
    video_paths = select_files(initial_dir)  # Открываем диалоговое окно для выбора видеофайлов
    if video_paths:
        save_folder = select_folder(initial_dir)  # Открываем диалоговое окно для выбора папки
        if save_folder:
            play_videos(video_paths, save_folder)
        else:
            print("No folder selected.")
    else:
        print("No videos selected.")
