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


def play_videos(video_path1, video_path2, save_folder1, save_folder2):
    frame_offset = 60

    cap1 = cv2.VideoCapture(video_path1)
    cap2 = cv2.VideoCapture(video_path2)

    if not cap1.isOpened() or not cap2.isOpened():
        print("Error: Could not open one of the videos.")
        return

    cv2.namedWindow('Videos', cv2.WINDOW_NORMAL)

    frame_count1 = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count2 = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_pos1 = 0
    frame_pos2 = 0

    while True:
        cap1.set(cv2.CAP_PROP_POS_FRAMES, frame_pos1)
        cap2.set(cv2.CAP_PROP_POS_FRAMES, frame_pos2)
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not ret1 or not ret2:
            print("Error: Could not read frame from one of the videos.")
            break

        combined_frame = cv2.hconcat([frame1, frame2])
        cv2.imshow('Videos', combined_frame)

        key = cv2.waitKey(0)

        if key == 27:  # ESC key to exit
            break
        elif key == 81 or key == 2424832:  # Left arrow key
            frame_pos1 = max(0, frame_pos1 - frame_offset)
            frame_pos2 = max(0, frame_pos2 - frame_offset)
        elif key == 83 or key == 2555904:  # Right arrow key
            frame_pos1 = min(frame_count1 - frame_offset, frame_pos1 + frame_offset)
            frame_pos2 = min(frame_count2 - frame_offset, frame_pos2 + frame_offset)
        elif key == 84 or key == 2621440:  # Down arrow key
            save_path1 = os.path.join(save_folder1, f'frame_{frame_pos1}.png')
            save_path2 = os.path.join(save_folder2, f'frame_{frame_pos2}.png')
            cv2.imwrite(save_path1, frame1)
            cv2.imwrite(save_path2, frame2)
            print(f'Saved frame {frame_pos1} from video 1 as {save_path1}')
            print(f'Saved frame {frame_pos2} from video 2 as {save_path2}')

    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    initial_dir = "~/Видео/"  # Укажите путь к вашей начальной директории
    print("Select the first video")
    video_path1 = select_file(initial_dir)  # Открываем диалоговое окно для выбора первого видеофайла
    if video_path1:
        print("Select the second video")
        video_path2 = select_file(initial_dir)  # Открываем диалоговое окно для выбора второго видеофайла
        if video_path2:
            print("Select folder to save frames from the first video")
            save_folder1 = select_folder(initial_dir)  # Открываем диалоговое окно для выбора папки для первого видео
            if save_folder1:
                print("Select folder to save frames from the second video")
                save_folder2 = select_folder(
                    initial_dir)  # Открываем диалоговое окно для выбора папки для второго видео
                if save_folder2:
                    play_videos(video_path1, video_path2, save_folder1, save_folder2)
                else:
                    print("No folder selected for the second video.")
            else:
                print("No folder selected for the first video.")
        else:
            print("No second video selected.")
    else:
        print("No first video selected.")
