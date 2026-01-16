# Step 1 :import the module
import cv2 
import rerun as rr

if __name__ == "__main__":
    rr.init("5x5 video", spawn=True)
    # Step 2: Read the video from specified path
    video_path=r"C:\Users\rafai\Videos\Γίνεται εγγραφή 2024-11-19 210302.mp4"
    cam = cv2.VideoCapture(video_path)

    # Step 3: Get the frames per second (fps) of the video
    fps = cam.get(cv2.CAP_PROP_FPS)
    print("Frames per second : ", fps)

    # Step 4: Find the total number of frames in the video
    frame_count = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Total number of frames : ", frame_count)

    # Step 5: Calculate the duration of the video in seconds
    duration = frame_count / fps
    print("Duration (in seconds) : ", duration)
    counter = 0
    
    #step 6:Read and display video frame-by-frame
    while True:
        ret, frame = cam.read()
        counter+=1
        rr.set_time("frame", sequence=counter)

        if not ret:
            break   # No more frames → end of video

        # cv2.imshow("Video", frame)
        rr.log(
            "video_frame",
            rr.Image(frame[:, :, ::-1]),  # Convert BGR to RGB
        )

        # Press Q to quit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Release the video capture object
    cam.release()
    cv2.destroyAllWindows()