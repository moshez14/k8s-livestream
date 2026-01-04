import datetime
import http.client
import json
import os
import re
import smtplib
import time
from collections import defaultdict, deque
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

import cv2
import numpy as np

from dotenv import load_dotenv
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
from twilio.rest import Client
from ultralytics import YOLO

# Load environment variables from .env file
# ✅ LOAD ONCE — GLOBAL
load_dotenv("/app/.env")
# Load .live second (overrides .env if same keys exist)

def stream_view(rtmp_source,camera_id):

    #rtmp_url = f'rtmp://os.environ.get("HOST"):1935/live_hls/{rtmp_source}'
    print(f"CAMERA ID={camera_id}")
    response = get_streamUrl(camera_id)
    resp = add_rtmpCode(camera_id,"dummy")
    camera_index = resp.text
    host = os.getenv("HOST")
    response = f"rtmp://{host}/show/{camera_index}"
    #response = f"rtsp://admin:hahn030569!@91.135.106.136:554/Streaming/channels/201"
    print (f"RESPONSE={response}")
    rtmp_url = response
    #
    # Retrieve camera and mission names and AI question
    #
    camera_name = get_cameraName(camera_id)
    missionName = mission_name(rtmp_source)
    ai_question = get_question(rtmp_source, camera_id)
    #
    silent_messages = False
    severity = "Normal"
    intervalTime = 20
    video_question = ""
    video_overlap_seconds = 0
    video_check_mode = "current"
    data = get_mission_comp(rtmp_source, camera_id)
    print(f"DATA={data}")
    # Check if the response indicates that the record was not found
    data = data or {}
    if data.get("error") == "Record not found":
        print("Record not found. Skipping extraction.")
    else:
        severity = data.get("severity")
        silent_messages = data.get("silentMessages")
        intervalTime = data.get("intervalTime")
        video_question = data.get("video_question")
        video_overlap_seconds = int(data.get("video_chunk_seconds") * (data.get('video_chunk_overlap_percent', 20) / 100))
        video_check_mode = data.get("video_check_mode")
        print("Severity:", severity)
        print("Silent Messages:", silent_messages)
        print("IntervalTime:", intervalTime)
        print("video_question:", video_question)
        print("video_overlap_seconds:", video_overlap_seconds)
        print("video_check_mode:", video_check_mode)
    #
    #rtmp_url = json.loads(json.loads(json.dumps(response.json()))['cameraDetails'])['streamUrl']
    print (f"RTMP={rtmp_url}")
    #rtmp_url = rtmp_source
    archive_path = "/var/www/html/show"
    #base_path = "/home/ubuntu/livestream/stream/data"
    #base_path = "/mnt/mp4-efs/dynamic_provisioning/pvc-f9f3c02e-bebd-40b0-b326-f22961bb1b14/stream/data"
    global prev_timestamp 
    global out, phone

    #global frame_count
    # Set the environment variable
    base_path = os.environ.get("BASE_PATH")
    os.environ['OPENCV_FFMPEG_READ_ATTEMPTS'] = '16384'
    #os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "timeout;5000000"
    FRAME_DIVIDE=int(os.environ.get('FRAME_DIVIDE'))
    CONFIDENCE=float(os.environ.get('CONFIDENCE'))
    OBJECT_DETECTED_COUNT = int(os.environ.get('OBJECT_DETECTED_COUNT'))
    next_send = 1
    object_model_details = model_objects(rtmp_source)

    object_model = object_model_details[0]['key']
    object_model_91 = any(obj["key"] == 91 or obj["key"] == 91.0 for obj in object_model_details)
    object_model_92 = any(obj["key"] == 92 or obj["key"] == 92.0 for obj in object_model_details)
    object_model_96 = any(obj["key"] == 96 or obj["key"] == 96.0 for obj in object_model_details)
    object_model_name =  object_model_details[0]['name']
    object_detect_model = object_model_details[0]['detectModel']

    #
    # Polygon values retrieval
    #
    print(f'Debug: object model {object_model} {object_model_name}')
    # Load the YOLOv8 model
    yolo_path = f"/app/yolov8/{object_detect_model}"
    model = YOLO(yolo_path)
    #model = YOLO('yolov8m-pose.pt')
    payload = ""
    headers = {
        'Authorization': 'Basic YWRtaW46QXVndV8yMDIz',
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'Accept-Encoding': 'gzip, deflate'
    }

    # Open the video file
    # video_path = f'rtmp://os.environ.get("HOST"):1935/live_hls/{rtmp}'
    print(rtmp_source)
    video_path = rtmp_url
    resp = add_rtmpCode(camera_id,"dummy")
    camera_index = resp.text
    #video_path = f"https://www.maifocus.com/show/{camera_index}"
    cap = cv2.VideoCapture(video_path)
    # Check if the video capture is successfully opened
    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 60000)  # 60 seconds
    cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 60000)  # 60 seconds
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)  # Reduce buffer to 3 frames

    track_history = defaultdict(lambda: [])
    track_history_alert = defaultdict(lambda: [])

    frame_count = -1 
    frame_false = -1
    # codec = cv2.VideoWriter_fourcc(*"mp4v")
    #codec = cv2.VideoWriter_fourcc(*"H264")

    #codec = cv2.VideoWriter_fourcc(*"AVC1")
    #codec = cv2.VideoWriter_fourcc(*"X264")
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    # save_path = str(Path("{}/process_{}_{}_{}".format(base_path,rtmp_source,camera_id,timestamp)).with_suffix('.mp4'))
    # save_path_mp4 = "copy_process_{}_{}_{}".format(rtmp_source, camera_id, timestamp)
    # new_save_path = save_path.replace("/process_", "/copy_process_")
    # new_save_path = str(Path("{}/copy_process_{}_{}".format(base_path, camera_index, timestamp)).with_suffix('.mp4'))
    # print (f"Save Path={save_path}")
    #file = open('/var/maifocus/maifocus.log', 'a')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # out = cv2.VideoWriter(save_path, codec, fps, (w, h))
    object_identifiers = {}
    object_not_identifiers = {}
    now = 0
    # Parameters
    motion_threshold = 3000  # Threshold for detecting motion. Changed this temporarily to -1 to disable motion detection
    motion_detected = True
    frame_count_motion = 0
    video_frame_count = 0
    max_video_frames = 3000
    video_part = 1
    consecutive_motion_count = 0# Initialize consecutive motion counter
    # Main loop to process video frames
    prev_frame = None
    last_frame_save_time = time.time()
    no_of_frames_to_skip = 3 # Number of frames to skip
    #current_video_file = None
    state = None

    #
    # Get polygon values
    #
    resp = get_polygon(rtmp_source, camera_id)
    polygon_coords = resp.get('polygon_coords', [])
    merged_polygon = unary_union([Polygon(polygon_coord) for polygon_coord in polygon_coords])
    polygon_coords_denormalized = [[(x * w, y * h) for x, y in polygon_coord] for polygon_coord in polygon_coords]

    #
    # Loop through the video frames
    #
    while cap.isOpened():
        for _ in range(no_of_frames_to_skip):  # Skip buffered frames
            cap.grab()


        # Read a frame from the video
        success, frame = cap.read()
        print(f"{camera_index} SUCCESS={success}")

        try:
            if success:
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
                frame_count = frame_count + 1
                #print (f'{camera_index} MISSION_ID={rtmp_source} FRAME COUNT={frame_count}')
                #timestamp2 = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
                #jpeg1 = str(Path("/var/www/html/show/frame_{}_{}_{}.jpg".format(rtmp_source, camera_id, timestamp2)))
                #print(f"{camera_index} timestamp={timestamp}")
               # cv2.imwrite(jpeg1, frame)

                #
                # A message whether a video is streaming or not is being sent
                # within a range of 5 minutes each 10 minutes.
                #
                # Release the video capture object and close the display window
                #
                # A mess#age whether a video is streaming or not is being sent
                # within a range of 5 minutes each 10 minutes.
                # frame_count % FRAME_DIVIDE
                if (frame_count % FRAME_DIVIDE) == 0 and get_mission_status(rtmp_source).lower()=='initiating':
                    now = datetime.datetime.now()
                    print("TIME RUNNING=", ceil_dt(now, datetime.timedelta(minutes=10)))
                    ceil_time = ceil_dt(now, datetime.timedelta(minutes=1))
                    range_start = ceil_time - datetime.timedelta(minutes=2)  # 2.5 minutes before
                    range_end = ceil_time + datetime.timedelta(minutes=2)
                    print(f'{camera_index} RANGE_START={range_start} RANGE_END={range_end}')
                    if range_start <= now <= range_end:
                       update_mission_status(rtmp_source, "running")

                ###meni add### 

                print("All numbers in object_model:", object_model)
                    # Initialize prev_frame on the first loop iteration
                if prev_frame is None:
                    prev_frame = frame.copy()
                    print(f"{camera_index} prev_frame is null")
                    continue

                save_path, save_path_mp4 = find_current_mp4_filename(
                    base_path, rtmp_source, camera_id, timestamp, video_question, video_overlap_seconds, video_check_mode
                )

                frame_count_motion +=1
                if object_model_91:
                    current_time = time.time()
                    print(f"{camera_index} Object model 91 detected,i intervalTime={intervalTime} current_time={current_time} last_frame_save_time={last_frame_save_time} ")
                    if current_time - last_frame_save_time >= intervalTime:  # 60 seconds = 1 minute
                        #new_save_path = save_path.replace("/process_", "/copy_process_")
                        print(f"{camera_index} model_91 timestamp={timestamp}")
                        saved_frame_path = save_motion_frame(frame, rtmp_source, camera_id, polygon_coords_denormalized, timestamp, save_path_mp4)
                        print(f"{camera_index} Object model 91 detected, saved frame at: {saved_frame_path}")
        
                        # Update the last save time to the current time
                        last_frame_save_time = current_time
                        #continue
                ####end of timing #####

                # Check for motion if not in the YOLOv8 processing phase
                if not motion_detected:
                    motion_detected = True

                # If motion is detected, run YOLOv8 for the next 50 frames
                if motion_detected:
                    if object_model_96:
                        if detect_motion(frame, prev_frame, polygon_coords_denormalized, motion_threshold):
                            consecutive_motion_count += 1
                            print(f"{camera_index} Motion detected at number of frame: {consecutive_motion_count}")
                            current_time = time.time()
                            if consecutive_motion_count >= 25 and (current_time - last_frame_save_time) >= 10:
                                #new_save_path = save_path.replace("/process_", "/copy_process_")
                                save_motion_frame(frame, rtmp_source, camera_id, polygon_coords_denormalized, timestamp, save_path_mp4)
                                print(f"{camera_index} saved frame after motion" )
                                print(f"{camera_index} time diff: {current_time - last_frame_save_time}")
                                consecutive_motion_count = 0
                                last_frame_save_time = current_time

                        else:
                            consecutive_motion_count = 0  # Reset if motion is not detected in consecutive frames

                    if object_model_92:
                        current_time = time.time()
                        print(f"{camera_index} Object model 96 detected, current_time={current_time}")
                        is_detected, state = detect_motion_full_frame(frame, prev_frame, fps, state, frame_count, camera_index, rtmp_source, camera_id)
                        if is_detected:
                            print(f"{camera_index} Object model 96 detected, saving frame")
                            saved_frame_path = save_motion_frame(frame, rtmp_source, camera_id, polygon_coords_denormalized, timestamp, save_path_mp4)
                            print(f"{camera_index} Saved frame at: {saved_frame_path}")
                            last_frame_save_time = current_time

                    prev_frame = frame.copy()
                    ### meni end###
                    # Run YOLOv8 inference on the frame
                    print(f"YOLOv8 running")
                    print(frame_count_motion)
                    results = model.track(frame, persist=True, conf=CONFIDENCE)
                    annotated_frame = results[0].plot()
                    for detection in range(len(results[0].boxes)):
                        detection = int(detection)
                        class_label = np.copy(results[0].boxes.cls.cpu())[detection]
                        track_id = results[0].boxes.id.int().cpu()[detection] if results[0].boxes.id is not None else -1
                        conf = np.copy(results[0].boxes.conf.cpu())[detection]
                        print(f"DETECTION={detection}")
                        print(f"CLASS LABELS={class_label}")
                        print(f"TRACK_ID={track_id}")
                        print(f"CONFIDENCE={conf}")
                        boxes = results[0].boxes.xywh.cpu()
                        # track_ids = results[0].boxes.id.int().cpu().tolist()
                        if results[0].boxes is not None and track_id != -1:
                            x, y, w, h = np.array(results[0].boxes.xywh.cpu())[detection]
                            payload = json.dumps({
                                "class_label": "{}".format(int(class_label)),
                                "object_model": "{}".format(int(object_model)),
                                "frame_count": frame_count,
                                "fps": fps,
                                "confidence": np.array(results[0].boxes.conf.cpu())[detection].tolist(),
                                "track_id": int(track_id),
                                "tensor_xywh": np.array(results[0].boxes.xywh.cpu())[detection].tolist(),
                                "tensor_xyxy": np.copy(results[0].boxes.xyxy.cpu())[detection].tolist(),
                                "tensor_xywhn": np.array(results[0].boxes.xywhn.cpu())[detection].tolist(),
                                "tensor_xyxyn": np.array(results[0].boxes.xyxyn.cpu())[detection].tolist(),
                                "video_source": video_path,
                                "camera_index": camera_index,
                                "video": save_path
                            })
                            # Visualize the results on the frame
                            print(f"payload={payload}")
                            track = track_history[int(track_id)]
                            track.append((float(x), float(y)))  # x, y center point
                            if len(track) > 30:  # retain 30 tracks for FRAME_DIVIDE frames
                                track.pop(0)

                            # Draw the tracking lines
                            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                            #cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)
                            #file.write(payload + "\n")
                        else:
                            payload = json.dumps({
                                "class_label": "{}".format(int(class_label)),
                                "object_model": "{}".format(int(object_model)),
                                "frame_count": frame_count,
                                "fps": fps,
                                "confidence": np.array(results[0].boxes.conf.cpu())[detection].tolist(),
                                "track_id": -1,
                                "tensor_xywh": np.array(results[0].boxes.xywh.cpu())[detection].tolist(),
                                "tensor_xyxy": np.copy(results[0].boxes.xyxy.cpu())[detection].tolist(),
                                "tensor_xywhn": np.array(results[0].boxes.xywhn.cpu())[detection].tolist(),
                                "tensor_xyxyn": np.array(results[0].boxes.xyxyn.cpu())[detection].tolist(),
                                "video_source": video_path,
                                "camera_index": camera_index,
                                "video": save_path
                            })
                            #file.write(payload + "\n")
                    #frame_count = frame_count + 1

                    print ("before write ")
                    # for _ in range(no_of_frames_to_skip + 1):
                    #     out.write(annotated_frame)

                    #out.write(frame)
                    print (f'{camera_index} CAMERA_ID={camera_id} MISSION_ID={rtmp_source} FRAME COUNT={frame_count} FRAME_DIVIDE={FRAME_DIVIDE}')
                    #
                    # Create a new mp4 output each time number of
                    # frames form a FRAME_DIVIDE (i.e. 2 minutes)  movie.
                    #
                    if (frame_count % FRAME_DIVIDE) == 0:
                        print(f"{camera_index} frame_count={frame_count} FRAME_DIVIDE={FRAME_DIVIDE}")
                        # out.release()
                        # print(f"{camera_index} after release")
                        # if track_history_alert or object_model_91 or object_model_92:
                        #     new_save_path = save_path.replace("/process_", "/copy_process_")
                        #     print("NEW NAME={} OLD NAME={}".format(new_save_path,save_path))
                        #     result = subprocess.Popen(["/usr/bin/mv", save_path, new_save_path], stdout=subprocess.PIPE,stderr=subprocess.PIPE, text=True)
                        #
                        #     # Check if the command was successful
                        #     print(f"result={result}")
                        #     if result.returncode == 0:
                        #         print(f"{camera_index} File moved from {save_path} to {new_save_path}")
                        #     else:
                        #         print(f"{camera_index} File failed to move {save_path} to {new_save_path}")
                        # print(f"{camera_index} track_history_alert={track_history_alert}")
                        # timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
                        # save_path = str(Path("{}/process_{}_{}_{}".format(base_path,rtmp_source,camera_id, timestamp)).with_suffix('.mp4'))
                        # save_path_mp4 = "copy_process_{}_{}_{}".format(rtmp_source,camera_id, timestamp)
                        # print (save_path)
                        # fps = int(cap.get(cv2.CAP_PROP_FPS))
                        # w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        # h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        # out = cv2.VideoWriter(save_path, codec, fps, (w, h))
                        print(f'{camera_index} frame count={frame_count}')
                        mission_status = get_mission_status(rtmp_source)
                        print (f"{camera_index} MISSION STATUS={mission_status}")

                        track_history_alert.clear()
                        ####
                        # Getting severity value
                        ####
                        #
                        silent_messages = False
                        severity = "Normal"
                        intervalTime = 20
                        video_question = ""
                        data = get_mission_comp(rtmp_source, camera_id)
                        data = data or {}
                        # Check if the response indicates that the record was not found
                        if data.get("error") == "Record not found":
                            print("Record not found. Skipping extraction.")
                        else:
                            severity = data.get("severity")
                            silent_messages = data.get("silentMessages")
                            intervalTime = data.get("intervalTime")
                            video_question = data.get("video_question")
                            video_overlap_seconds = int(data.get("video_chunk_seconds") * (data.get('video_chunk_overlap_percent', 20) / 100))
                            video_check_mode = data.get("video_check_mode")
                            print("Severity:", severity)
                            print("Silent Messages:", silent_messages)
                            print("IntervalTime:", intervalTime)
                            print("video_question:", video_question)
                            print("video_overlap_seconds:", video_overlap_seconds)
                            print("video_check_mode:", video_check_mode)
                        #
                        ####
                        object_model_details = model_objects(rtmp_source)
                        object_model_91 = any(obj["key"] == 91 or obj["key"] == 91.0 for obj in object_model_details)
                        object_model_92 = any(obj["key"] == 92 or obj["key"] == 92.0 for obj in object_model_details)

                        ai_question = get_question(rtmp_source, camera_id)

                        resp = get_polygon(rtmp_source, camera_id)
                        polygon_coords = resp.get('polygon_coords', [])
                        merged_polygon = unary_union([Polygon(polygon_coord) for polygon_coord in polygon_coords])
                        polygon_coords_denormalized = [[(x * w, y * h) for x, y in polygon_coord] for polygon_coord in polygon_coords]

                        if str(mission_status).lower()=='stopped' or str(mission_status).lower()=='upcoming' or str(mission_status).lower()=='paused':
                            exit()

                    print(f'{camera_index} len(results[0].boxes)={len(results[0].boxes)}')

                    #
                    # For each detection in frame
                    #
                    for detection in range(len(results[0].boxes)):
                        detection = int(detection)
                        class_label = results[0].boxes.cls[detection]
                        confidence = np.array(results[0].boxes.conf.cpu())[detection].tolist()


                        track_id = results[0].boxes.id.int().cpu()[detection] if results[0].boxes.id is not None else -1
                        print(f"{camera_index} TRACK_ID1={track_id}")

                        print(f'{camera_index} TRACK_HISTORY_ALERT={track_history_alert[int(track_id)]}')

                        tensor_xyxyn = np.array(results[0].boxes.xyxyn.cpu())[detection]
                        x_min_tensor = tensor_xyxyn[0]
                        y_min_tensor = tensor_xyxyn[1]
                        x_max_tensor = tensor_xyxyn[2]
                        y_max_tensor = tensor_xyxyn[3]

                        # Compute center of the bounding box
                        cx = (x_min_tensor + x_max_tensor) / 2
                        cy = (y_min_tensor + y_max_tensor) / 2
                        pt = Point(cx, cy)

                        # Check if no polygons selected or the point lies within polygons
                        object_detected_in_polygon = len(polygon_coords) < 1 or merged_polygon.contains(pt)

                        print(f'{camera_index} detection={detection} tensor_xyxyn = {tensor_xyxyn} x_min_tensor={x_min_tensor} y_min_tensor={y_min_tensor} x_max_tensor={x_max_tensor} y_max_tensor={y_max_tensor} object_detected_in_polygon={object_detected_in_polygon} pt={pt} polygon_coords={polygon_coords} polygon_coords_denormalized={polygon_coords_denormalized}')
                        if object_detected_in_polygon:
                            #
                            # Having the dictionary track_history_alert being filled with timestamps
                            # that track was detected with.
                            #
                            #
                            # track is being followed only
                            # if it is inside the polygon.
                            #
                            track_alert = track_history_alert[int(track_id)]
                            track_alert.append(datetime.datetime.now())
                            print(f'{camera_index} detection={detection} tensor_xyxyn = {tensor_xyxyn}')
                            for model_count in range(len(object_model_details)):
                                timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
                                print(f'{camera_index} model_count={model_count} object_model_details={object_model_details} mode_count={model_count}')
                                object_model = object_model_details[model_count]['key']
                                object_model_name = object_model_details[model_count]['name']
                                print(f'{camera_index} class label={class_label} object_model={object_model} confidence={confidence}')
                                if class_label == object_model and confidence > CONFIDENCE:
                                    #
                                    # Check if the model has an entry in the dictionaries, if not, initialize them
                                    #
                                    if object_model not in object_identifiers:
                                        object_identifiers[object_model] = 0
                                        object_not_identifiers[object_model] = 0
                                        print(f'{camera_index}  object_model={object_model}')


                                    # Increment the object identifier for the current model
                                    print(f"Object Identifiers={object_identifiers}")
                                    object_identifiers[object_model] += 1
                                    obj_count = object_identifiers[object_model]

                                    print(
                                        f'{camera_index} frame count {frame_count} class_label {class_label} model {object_model} obj_count {obj_count} Object Model Name {object_model_name}')

                                    #
                                    # For each detected object, try to see if it is over threshold objects detected
                                    # If object detected is larger then the number of objects requested for the
                                    # amount of frames, or the object has been initially detected, then
                                    # Send an alert
                                    found_track_id = 0
                                    print(f'{camera_index} LEN1_OF_HISTORY={len(track_history_alert[int(track_id)])}')
                                    #
                                    # If this is the third time a track id is identified
                                    #

                                    if len(track_history_alert[int(track_id)]) == OBJECT_DETECTED_COUNT:
                                        found_track_id = 1
                                    else:
                                        found_track_id = 0
                                    #elif len(track_history_alert[int(track_id)]) == int(FRAME_DIVIDE/2):
                                    #     found_track_id = 1
                                    #elif len(track_history_alert[int(track_id)]) == (FRAME_DIVIDE-1):
                                    #    found_track_id = 1


                                    if found_track_id > 0:
                                        # Save the current movie for the message to come
                                        # new_save_path = save_path.replace("/process_", "/copy_process_")
                                        print("NEW NAME IN ALERT={}".format(save_path))
                                        #
                                        # If it is over the threshold, then use the following formula
                                        # in order to figure out the next time an SMS should
                                        # be send.
                                        #
                                        #os.rename(save_path,new_save_path)
                                        jpeg = str(Path("/var/www/html/show/frame_{}_{}_{}.jpg".format(rtmp_source, camera_id, timestamp)))
                                        cv2.imwrite(jpeg, annotated_frame)
                                        #if video_question:
                                        jpeg_mp4 = str(Path("/var/www/html/show/frame_{}_{}_{}_{}.jpg".format(rtmp_source, camera_id, timestamp,save_path_mp4)))
                                        print(f"{camera_index} jpeg_mp4={jpeg_mp4}")
                                        cv2.imwrite(jpeg_mp4, annotated_frame)
                                        jpeg_masked = str(Path("/var/www/html/show/frame_{}_{}_{}_masked.jpg".format(rtmp_source, camera_id, timestamp)))
                                        annotated_frame_masked = annotated_frame
                                        if polygon_coords_denormalized:
                                            mask = np.zeros_like(annotated_frame, dtype=np.uint8)
                                            for poly in polygon_coords_denormalized:
                                                cv2.fillPoly(mask, np.array([list(poly)], dtype=np.int32), (255, 255, 255))
                                            annotated_frame_masked = cv2.bitwise_and(annotated_frame, mask)
                                        cv2.imwrite(jpeg_masked, annotated_frame_masked)

                                        print("BEFORE RAW")

                                        raw_url = "https://{}/show/{}".format(os.environ.get("HOST"),save_path.split("/")[-1])
                                        raw_url_jpg_orig = "https://{}/show/frame_{}_{}_{}.jpg".format(os.environ.get("HOST"),rtmp_source, camera_id, timestamp)
                                        print("RAW URL={}".format(raw_url))
                                        print("RAW URL ORIG={}".format(raw_url_jpg_orig))
                                        #short_url = gain_short_url(raw_url)
                                        short_url = raw_url
                                        #raw_url_jpg = gain_short_url(raw_url_jpg_orig)
                                        raw_url_jpg = raw_url_jpg_orig
                                        message_to_send = "Maifocus {} detected: \nMission: {} \nCamera {}  \nPhoto Link:  {} .\nVideo will be shortly available at {}\n \nMAi Focus team \nsupport@mai-focus.com".format(
                                            object_model_name, missionName, camera_name, raw_url_jpg, short_url)
                                        print(f"SHORT URL={short_url}")
                                        print(f"SHORT URL JPG={raw_url_jpg}")
                                        print(
                                            f'{camera_index} frame count {frame_count} model {object_model} has arrived {obj_count} times in the last {FRAME_DIVIDE} frames')
                                        object_identifiers[object_model] = 0
                                        user_info = mission_details(rtmp_source)
                                        print(f'{camera_index} USER_INFO1={user_info} len(user_info)={len(user_info)}')

                                    #
                                    # Check if AI question is present or not.
                                    # If it is present, don't send an SMS and email.
                                    #
                                        if ai_question == "Default":
                                            # ---
                                            # Phone and email retrieval
                                            # ---

                                            email = []
                                            phone = []

                                            if user_info and len(user_info) > 0:
                                                #
                                                # In case a mission id defined with just
                                                # an sms for example
                                                #
                                                if user_info[0].get('type') == 'email' and 'value' in user_info[0]:
                                                    email = user_info[0]['value']
                                                    print(f'{camera_index} Email number exists: {email}')
                                                else:
                                                    print(f'{camera_index} Email does not exists {email}')
                                                if user_info[0].get('type') == 'sms' and 'value' in user_info[0]:
                                                    phone = user_info[0]['value']
                                                    print(f'{camera_index} Phone number exists: {phone}')
                                                else:
                                                    print(f'{camera_index} Phone does not exists {phone}')

                                            # else:
                                            #     user_info = user_details(rtmp_source)
                                            #     print(f'USER_INFO_2={user_info}')
                                            #     email = [user_info['email']]

                                            # user_info = mission_details(rtmp_source)
                                            if user_info and len(user_info) > 1:
                                                if user_info[0].get('type') == 'sms' and 'value' in user_info[0]:
                                                    phone = user_info[0]['value']
                                                    print(f'{camera_index} 1Phone number exists: {phone}')
                                                elif user_info[1].get('type') == 'sms' and 'value' in user_info[1]:
                                                    phone = user_info[1]['value']
                                                    print(f'{camera_index} 2Phone number exists: {phone}')
                                                else:
                                                    print(f'{camera_index} Phone number does not exist {phone}.')
                                            # else:
                                            #     user_info = user_details(rtmp_source)
                                            #     print(f'USER_INFO_PHONE={user_info}')
                                            #     phone = [user_info['phone']]
                                            for em in email:
                                                print(f'{camera_index} Email={em}')
                                                if em is None:
                                                    em = "maifocus12@gmail.com"
                                                #send_email(em,"Alert Detected: {} is detected at camera {} in mission {} . Login into {}:8080/show/process_{}_{}.mp4".format(object_model_name, camera_name, missionName, os.environ.get("HOST"), camera_index, prev_timestamp))
                                                #send_email(em,"Alert Detected: {} is detected at camera {} in mission {} . Picture is at {}:8080/show/frame_{}_{}_{}.jpg . Video will be shortly available at {}:8080/show/{}".format(object_model_name, camera_name, missionName, os.environ.get("HOST"), rtmp_source, camera_id, timestamp, os.environ.get("HOST"),save_path.split("/")[-1]))
                                                send_email(em,message_to_send)

                                            print(f'{camera_index} PHONE={phone}')
                                            if phone and len(phone) > 0:
                                                for sms in phone:
                                                    print(f'{camera_index} sms={sms}')
                                                    international_phone = convert_to_international(sms)
                                                    #send_sms_pulsim(international_phone," Alert Detected: {} detected at camera {} in mission {}. Login into {}:8080/show/process_{}_{}.mp4".format(object_model_name, camera_name, missionName, os.environ.get("HOST"), camera_index, prev_timestamp))
                                                    if (silent_messages == False):
                                                        send_sms_pulsim(international_phone,message_to_send)

                                            if (severity == "Normal"):
                                                update_notification(rtmp_source,camera_id,message_to_send,raw_url_jpg, short_url,"/notificationSounds/regular-cam.mp3")
                                            else:
                                                update_notification(rtmp_source, camera_id, message_to_send, raw_url_jpg,
                                                               short_url, "/notificationSounds/attention-cam.mp3")
                                        #update_video(rtmp_source,camera_id,"Maifocus {} detected: \nMission: {} \nCamera: {}  \nPhoto Link:  {} .\nVideo will be shortly available at {}\n \nMAi Focus team \nsupport@mai-focus.com".format(object_model_name,missionName, camera_name, camera_index,timestamp,raw_url_jpg, os.environ.get("HOST"),save_path.split("/")[-1]),"{}".format(short_url))
                ####meniadd### Update prev_frame for the next iteration
                   # frame_count_motion += 1
                  #  if frame_count_motion >= 50:
                 #       motion_detected = False
                #        print("Finished running YOLOv8 on 50 frames, resuming motion detection.")
               # prev_frame = frame.copy()
                ####meniend###

            else:   # if success == False
                frame_false = frame_false + 1
                print(f"{camera_index} FRAME FALSE={frame_false}")
                #
                # A message whether a video is streaming or not is being sent
                # within a range of 1 minutes each 30 minutes.
                #
                #if (frame_false % FRAME_DIVIDE) == 0:
                    #update_message(rtmp_source,camera_id, " No video for camera {} in mission {}. ".format(camera_name, missionName)," "," ")

                now = datetime.datetime.now()
                print("TIME=",ceil_dt(now, datetime.timedelta(minutes=30)))
                ceil_time = ceil_dt(now, datetime.timedelta(minutes=30))
                range_start = ceil_time - datetime.timedelta(seconds=2)  # 2.5 minutes before
                range_end = ceil_time + datetime.timedelta(seconds=2)
                if range_start <= now <= range_end:
                    update_notification(rtmp_source,camera_id, " No video for camera {} in mission {}. ".format(camera_name, missionName)," "," ","attention-cam.mp3")
                    print(f'{camera_index} RTMP SOURCE={rtmp_source}')
                    update_mission_status(rtmp_source, "initiating")
                    user_info = mission_details(rtmp_source)
                    # ---
                    # Phone and email retrieval
                    # ---
                    email = None
                    print(f'{camera_index} USER_INFO={user_info}')

                    if user_info and len(user_info) > 0:
                        if user_info[0].get('type') == 'email' and 'value' in user_info[0]:
                            email = user_info[0]['value']
                            print(f'{camera_index} Email number exists: {email}')
                        else:
                            print(f'{camera_index} Email does not exists')
                    else:
                        user_info = user_details(rtmp_source)
                        print(f'{camera_index} USER_INFO={user_info}')
                        email = [user_info['email']]

                    phone = None
                    user_info = mission_details(rtmp_source)
                    print(f'{camera_index} USER_INFO3={user_info}')
                    if user_info and len(user_info) > 1:
                        if user_info[0].get('type') == 'sms' and 'value' in user_info[0]:
                            phone = user_info[0]['value']
                            print("Phone number exists:", phone)
                            print(f'{camera_index} 21Phone number exists: {phone}')
                        elif user_info[1].get('type') == 'sms' and 'value' in user_info[1]:
                            phone = user_info[1]['value']
                            print(f'{camera_index} 22Phone number exists: {phone}')
                        else:
                            print("Phone number does not exist.")
                    else:
                        user_info = user_details(rtmp_source)
                        phone = [user_info['phone']]
                #else:
                    #print("Either user_info is None or user_info[1] does not exist.")

                    for em in email:
                        print(f'{camera_index} EmailE={em}')
                        if em is None:
                            em = "maifocus12@gmail.com"
                        send_email(em," No video for camera {} in mission {}. ".format(camera_name, missionName))
                    for sms in phone:
                        print(f'{camera_index} SmsE={sms}')
                        print(sms)
                        international_phone = convert_to_international(sms)
                        #send_sms_pulsim(international_phone,
                        #           " No video for camera {} in mission {}. Camera source: {} -- Code {}".format(camera_name, missionName, video_path,camera_index))

                    mission_status = get_mission_status(rtmp_source)
                    print (f"{camera_index} MISSION STATUS BEFORE INIT={mission_status}")
                    if str(mission_status).lower()=='stopped' or str(mission_status).lower()=='upcoming' or str(mission_status).lower()=='paused' or str(mission_status).lower()=='running':
                        exit()
                    #update_mission_status(rtmp_source, "initiating")
                time.sleep(30)
                # else:
                #     print("The time doesn't match the ceiling time.")
        except Exception as e:
            # Handle the exception
            print(f"{camera_index} An error occurred: {frame_count} Exception={e}")
            # Optionally, you can log the error or perform other actions
            continue  # Continue to the next iteration of the loop
    return "OK"


def add_rtmpCode(id,rtmpCode):
    url = f"http://{os.environ.get('HOST')}:5500/api/addRtmpCode"
    payload = json.dumps({
        "id": str(id),
        "rtmpCode": rtmpCode
    })
    headers = {
        'Content-Type': 'application/json-patch+json',
        'Authorization': 'Basic YWRtaW46QXVndV8yMDIz'
    }
    print(f"Payload = {payload}")
    response = requests.request("POST", url, headers=headers, data=payload)
    return response

def get_streamUrl(camera_id):
    url = f"http://{os.environ.get('HOST')}:5500/api/get_camera_by_stream/"
    print(f"streamUrl host={os.environ.get('HOST')}")
    payload = json.dumps({
        "camera_id": camera_id
    })
    headers = {
        'Content-Type': 'application/json-patch+json',
        'Authorization': 'Basic YWRtaW46QXVndV8yMDIz'
    }

    response = requests.request("GET", url, headers=headers, data=payload)
    return (json.loads(json.loads(json.dumps(response.json()))['cameraDetails'])['streamUrl'])

def get_cameraName(camera_id):
    url = f"http://{os.environ.get('HOST')}:5500/api/get_camera_by_stream/"
    payload = json.dumps({
        "camera_id": camera_id
    })
    headers = {
        'Content-Type': 'application/json-patch+json',
        'Authorization': 'Basic YWRtaW46QXVndV8yMDIz'
    }

    response = requests.request("GET", url, headers=headers, data=payload)
    return (json.loads(json.loads(json.dumps(response.json()))['cameraDetails'])['name'])

def get_polygon(mission_id,camera_id):

  url = f"http://{os.environ.get('HOST')}:5500/api/get_polygon/{mission_id}/{camera_id}"

  payload = ""
  headers = {
    'Content-Type': 'application/json-patch+json',
    'Authorization': 'Basic YWRtaW46QXVndV8yMDIz'
  }

  response = requests.request("GET", url, headers=headers, data=payload)

  return response.json()

def convert_to_international(phone_number):
    # Check if the phone number starts with '0' (indicating it's a local number)
    if phone_number.startswith('0'):
        # Remove the leading '0' and add the international dialing code for Israel
        international_number = '+972' + phone_number[1:]
        return international_number
    else:
        # The input is already in international format
        return phone_number

def user_details(mission_id):
    import requests
    import json

    url = f"http://{os.environ.get('HOST')}:5500/api/userDet/"
    payload = json.dumps({
        "mission_id": mission_id
    })
    headers = {
        'Content-Type': 'application/json-patch+json',
        'Authorization': 'Basic YWRtaW46QXVndV8yMDIz'
    }

    response = requests.request("GET", url, headers=headers, data=payload)
    return (response.json()['userDetails'])

def mission_details(mission_id):
    import requests
    import json

    url = f"http://{os.environ.get('HOST')}:5500/api/userDet/"
    payload = json.dumps({
        "mission_id": mission_id
    })
    headers = {
        'Content-Type': 'application/json-patch+json',
        'Authorization': 'Basic YWRtaW46QXVndV8yMDIz'
    }

    response = requests.request("GET", url, headers=headers, data=payload)
    return (response.json()['missionDetails'])

def mission_name(mission_id):
    import requests
    import json

    url = f"http://{os.environ.get('HOST')}:5500/api/userDet/"
    payload = json.dumps({
        "mission_id": mission_id
    })
    headers = {
        'Content-Type': 'application/json-patch+json',
        'Authorization': 'Basic YWRtaW46QXVndV8yMDIz'
    }

    response = requests.request("GET", url, headers=headers, data=payload)
    return (response.json()['missionName'])

def model_objects(mission_id):
    import requests
    import json

    url = f"http://{os.environ.get('HOST')}:5500/api/userDet/"
    payload = json.dumps({
        "mission_id": mission_id 
    })
    headers = {
        'Content-Type': 'application/json-patch+json',
        'Authorization': 'Basic YWRtaW46QXVndV8yMDIz'
    }

    response = requests.request("GET", url, headers=headers, data=payload)
    if response:
        return (json.loads(json.loads(json.dumps(response.json()))['detectObjects']))

def update_message(rtmp_code,camera_id,message,photo_url,video_url,notification):
    import requests
    import json

    print(f'UPDATE MESSAGE {message}')
    url = f"http://{os.environ.get('HOST')}:5500/api/messagelog"
    payload = json.dumps({
        "mission_id": rtmp_code,
        "camera_id": camera_id,
        "message": message,
        "notification": notification,
        "photo_url": photo_url,
        "video_url": video_url
    })
    headers = {
        'Content-Type': 'application/json-patch+json',
        'Authorization': 'Basic YWRtaW46QXVndV8yMDIz'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    print(response.text)

def update_notification(rtmp_code,camera_id,message,photo_url,video_url,notification):
    import requests
    import json

    print(f'UPDATE MESSAGE {message}')
    url = f"http://{os.environ.get('HOST')}:5500/api/lognotification"
    payload = json.dumps({
        "mission_id": rtmp_code,
        "camera_id": camera_id,
        "message": message,
        "photo_url": photo_url,
        "video_url": video_url,
        "notification": notification
    })
    headers = {
        'Content-Type': 'application/json-patch+json',
        'Authorization': 'Basic YWRtaW46QXVndV8yMDIz'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    print(response.text)

def update_message_comp(missionID, cameraID, severity, silentMessages):
    # """
    # Calls the REST API to update or insert a record in the missionMessages collection.
    #
    # Parameters:
    # - missionID (str): The missionID as a string.
    # - cameraID (str): The cameraID as a string.
    # - severity (str): The severity value ("Normal" or "Critical").
    # - silentMessages (bool): True or False.
    # - url (str): The API endpoint URL.
    #
    # Returns:
    # - dict: The JSON response from the API if the request was successful.
    # - None: In case of an error.
    # """
    payload = {
        "missionID": missionID,
        "cameraID": cameraID,
        "severity": severity,
        "silentMessages": silentMessages
    }
    url = f"http://{os.environ.get('HOST')}:5500/update_mission_comp"
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()  # Will raise an HTTPError for bad responses (4XX or 5XX)
        return response.json()
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err} - Response: {response.text}")
    except Exception as err:
        print(f"An error occurred: {err}")

    return None


import requests


def get_mission_comp(missionID, cameraID):
    # """
    # Calls the REST API to read a record from the missionMessages collection based on missionID and cameraID.
    #
    # Parameters:
    # - missionID (str): The mission ID as a string.
    # - cameraID (str): The camera ID as a string.
    # - url (str): The API endpoint URL.
    #
    # Returns:
    # - dict: The JSON response from the API if the request was successful.
    # - None: In case of an error.
    # """
    params = {
        "missionID": missionID,
        "cameraID": cameraID
    }

    try:
        url = f"http://{os.environ.get('HOST')}:5500/get_mission_comp"
        response = requests.get(url, params=params)
        response.raise_for_status()  # Raise an error for bad responses
        return response.json()
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err} - {response.text}")
    except Exception as err:
        print(f"An error occurred: {err}")

    return None


def update_video(rtmp_code,camera_id,message,url_video):
    import requests
    import json

    url = f"http://{os.environ.get('HOST')}:5500/api/missionvideo"
    payload = json.dumps({
        "mission_id" : rtmp_code,
        "camera_id": camera_id,
        "message": message,
        "url_video": url_video
    })
    headers = {
        'Content-Type': 'application/json-patch+json',
        'Authorization': 'Basic YWRtaW46QXVndV8yMDIz'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    print(response.text)

def update_mission_status(mission_id,status):
    import requests
    import json

    url = f"http://{os.environ.get('HOST')}:5500/api/updatemission/{mission_id}/{status}"
    payload = json.dumps({})
    headers = {
        'Content-Type': 'application/json-patch+json',
        'Authorization': 'Basic YWRtaW46QXVndV8yMDIz'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    print(response.text)

def get_mission_status(mission_id):
    import requests
    import json

    url_mission = "http://{os.environ.get('HOST')}:5500/api/mission/"
    mission_id_json = str(mission_id)
    payload_mission = json.dumps({
        "mission_id": mission_id_json
    })
    print(f"Mission Payload={payload_mission}")
    headers = {
        'Content-Type': 'application/json-patch+json',
        'Authorization': 'Basic YWRtaW46QXVndV8yMDIz'
    }

    response_mission = requests.request("GET", url_mission, headers=headers, data=payload_mission)
    if response_mission.status_code == 200:
        return json.loads(json.loads(response_mission.text)['missionDetails'])[0]['status']

def gain_short_url(url):
    print("IN gain_short_url={}".format(url))
    conn = http.client.HTTPSConnection("shorturl9.p.rapidapi.com")

    payload = f"-----011000010111000001101001\r\nContent-Disposition: form-data; name=\"url\"\r\n\r\n{url}\r\n-----011000010111000001101001--\r\n\r\n"
    print(f"PAYLOAD={payload}")

    x_rappidapi_key = os.environ.get("X_RAPPIDAPI_KEY")
    headers = {
        'x-rapidapi-key': x_rappidapi_key,
        'x-rapidapi-host': "shorturl9.p.rapidapi.com",
        'Content-Type': "multipart/form-data; boundary=---011000010111000001101001"
    }

    conn.request("POST", "/functions/api.php", payload, headers)

    res = conn.getresponse()
    data = res.read()
    new_url = json.loads(data.decode("utf-8"))['url']
    print(f"NEW URL={new_url}")
    return new_url


def send_sms(number,object_detected):
    account_sid = os.environ.get("TWAccountSID")
    auth_token = os.environ.get("TWToken")

    client = Client(account_sid, auth_token)

    message = client.messages.create(
        to=number,
        from_=os.environ.get("TWPhone"),
        body="{}".format(object_detected))


def send_sms_pulsim(number,object_detected):
    url = "https://api.pulseem.com/api/v1/SmsApi/SendSms"

    payload = json.dumps({
        "sendId": "Maifocus",
        "isAsync": True,
        "cbkUrl": {os.environ.get('HOST')},
        "smsSendData": {
            "fromNumber": os.environ.get("PULPhone"),
            "toNumberList": [
                number
            ],
            "referenceList": [
                "null"
            ],
            "textList": [
                object_detected
            ],
            "sendTime": "",
            "isAutomaticUnsubscribeLink": True
        }
    })
    headers = {
        'Content-Type': 'application/json-patch+json',
        'x-api-key': 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c',
        'APIKey': os.environ.get("PULAPIKey"),
        'Authorization': 'Basic YWRtaW46QXVndV8yMDIz'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    print(response.text)

def send_email(email,object_detected):
    """Example endpoint returning a list of by email
     This is using docstrings for specifications.
     ---
tags:
  - Send an Email
parameters:
       - name: email
         in: path
         type: string
         required: true
         default: all
       - name: object_detected
         in: path
         type: string
         required: true
         default: all
responses:
       200:
         description: email was send
     """
    '''
    post smtp email
    '''
    # Email details
    # Create a multipart message
    reciever_address = email
    smtp_username = os.environ.get("SMTPUserName") 
    smtp_password = os.environ.get("SMTPPass")

    message = MIMEMultipart()
    #body = 'This is the body of the email.'
    #body = MIMEText(body)  # convert the body to a MIME compatible string
    #message.attach(body)
    first_three_words = ' '.join(object_detected.split()[:3])
    message["From"] = smtp_username
    message["To"] = reciever_address
    message["Subject"] = "MaiFocus: ALERT: {}".format(first_three_words)
    message.attach(MIMEText(object_detected, 'plain'))

    #message = f"""From: MaiFocus
    #To: To {email}  <to@todomain.com>
    #Subject: MaiFocus Alert Detected
    #{object_detected}
    #"""

    # with smtplib.SMTP(smtp_server, smtp_port) as server:
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", port=465) as server:
            server.login(smtp_username, smtp_password)
            server.sendmail(smtp_username, reciever_address, message.as_string())
            server.quit()
            print("Email sent successfully.")

    except smtplib.SMTPException:
           print("Error: unable to send email")
    return "Send"

def get_question(mission_id, camera_id):
    import requests
    import json

    url = f"http://{os.environ.get('HOST')}:5500/api/get_chat/{mission_id}/{camera_id}"
    payload = ""
    headers = {
        'Content-Type': 'application/json-patch+json',
        'Authorization': 'Basic YWRtaW46QXVndV8yMDIz'
    }

    response = requests.request("GET", url, headers=headers, data=payload)

    json_dict = json.loads(response.text)

    # Convert the dictionary back to a JSON string
    json_formatted_string = json.dumps(json_dict)
    response_dict = json.loads(json_formatted_string)
    # Access the 'question' key
    question = response_dict['question']
    return question

    # Function to detect motion
def detect_motion(frame, prev_frame, polygon_coords_denormalized, threshold):
    # Convert frames to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    # Create a mask for the merged shape
    mask = np.zeros_like(gray_frame, dtype=np.uint8)

    if polygon_coords_denormalized:
        for polygon_coord in polygon_coords_denormalized:
            polygon_points = np.array([list(polygon_coord)], dtype=np.int32)
            cv2.fillPoly(mask, polygon_points, (255, 255, 255))
    else:
        # If no polygons, consider the whole frame
        mask.fill(255)

    # Apply the mask to the frames
    masked_frame = cv2.bitwise_and(gray_frame, mask)
    masked_prev_frame = cv2.bitwise_and(gray_prev_frame, mask)

    # Compute the absolute difference between the current and previous frames
    diff = cv2.absdiff(masked_prev_frame, masked_frame)

    # Apply threshold to detect motion areas
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

    # Count the number of non-zero pixels (motion pixels)
    motion_pixels = cv2.countNonZero(thresh)
    print(f"the motion pixels is:{motion_pixels}")
    # If motion pixels exceed the threshold, return True (motion detected)
    return motion_pixels > threshold
    
def save_motion_frame(frame, rtmp_source, camera_id, polygon_coords_denormalized, timestamp, save_path_mp4):
    jpeg_path = str(Path("/var/www/html/show/frame_{}_{}_{}.jpg".format(rtmp_source, camera_id, timestamp)))
    cv2.imwrite(jpeg_path, frame)
    print(f"Saved motion frame to: {jpeg_path}")
    #if video_question:
    jpeg_mp4 = str(Path("/var/www/html/show/frame_{}_{}_{}_{}.jpg".format(rtmp_source, camera_id, timestamp, save_path_mp4)))
    print(f"{camera_id} jpeg_mp4={jpeg_mp4}")
    cv2.imwrite(jpeg_mp4, frame)
    frame_masked = frame
    jpeg_masked_path = str(Path("/var/www/html/show/frame_{}_{}_{}_masked.jpg".format(rtmp_source, camera_id, timestamp)))
    if polygon_coords_denormalized:
        mask = np.zeros_like(frame, dtype=np.uint8)
        for poly in polygon_coords_denormalized:
            cv2.fillPoly(mask, np.array([list(poly)], dtype=np.int32), (255, 255, 255))
        frame_masked = cv2.bitwise_and(frame, mask)
    cv2.imwrite(jpeg_masked_path, frame_masked)
    print(f"Saved masked motion frame to: {jpeg_masked_path}")
    return jpeg_path

def find_current_mp4_filename(base_path: str, rtmp_source: str, camera_id: str, current_timestamp: str, video_question: str, video_overlap_seconds: int, video_check_mode: str):
    current_timestamp_parsed = datetime.datetime.strptime(current_timestamp, "%Y-%m-%d-%H:%M:%S").replace(tzinfo=datetime.timezone.utc)
    current_file = None
    previous_file = None
    pattern = re.compile(f"^(process)_{rtmp_source}_{camera_id}.*\.mp4$")
    matching_files = []
    with os.scandir(base_path) as entries:
        for entry in entries:
            #print(f"MP4 {rtmp_source}_{camera_id} Entry={entry}")
            #print(f"MP4 entry.is_file={entry.is_file()} pattern.match={pattern.match(entry.name)}")
            if entry.is_file() and pattern.match(entry.name):
                date_string = entry.name.split('_')[-1].replace(".mp4", "")
                #print(f"MP$ date_string={date_string}")
                file_timestamp = datetime.datetime.strptime(date_string, "%Y-%m-%d-%H:%M:%S").replace(
                    tzinfo=datetime.timezone.utc)
                #print(f"MP$ file_stamp={file_timestamp}")
                matching_files.append((entry.name, file_timestamp))

    matching_files = sorted(matching_files, key=lambda x: x[1], reverse=True)
    #print(f"MP4 matching_file={matching_files}")
    for idx, (filename, timestamp) in enumerate(matching_files):
        if current_timestamp_parsed > timestamp:
            current_file = filename
            previous_file = matching_files[idx + 1][0] if idx + 1 < len(matching_files) else None
            break

    if not current_file:
        raise ValueError("No valid video file found in the specified directory.")

    save_path = str(Path("{}/{}".format(base_path, current_file)))
    save_path_mp4 = current_file

    date_string = current_file.split('_')[-1].replace(".mp4", "")
    timestamp = datetime.datetime.strptime(date_string, "%Y-%m-%d-%H:%M:%S").replace(tzinfo=datetime.timezone.utc)
    if video_question and previous_file and current_timestamp_parsed - timestamp <= datetime.timedelta(seconds=video_overlap_seconds):
        if video_check_mode == "old":
            save_path = str(Path("{}/{}".format(base_path, previous_file)))
            save_path_mp4 = previous_file
        elif video_check_mode == "both":
            save_path_mp4 = f"{previous_file}&{current_file}"

    save_path_mp4 = save_path_mp4.replace(".mp4", "")
    return save_path, save_path_mp4

N_ROWS=30
N_COLS=30
THRESH_BINARY=25
MORPH_KERNEL_SIZE=3
SAMPLE_FPS=2
REF_FRAMES=10
MAX_HISTORY_WINDOWS=30
TRIGGER_WINDOW_FRAMES=5
TRIGGER_STREAK=2
DELTA_RATIO=0.5
ABS_DELTA_MIN=10
COLOR_CHANGE_PCT=5.5
OUTPUT_FOLDER="/home/ubuntu/livestream/motion_frames"

def detect_motion_full_frame(
    frame, prev_frame, fps, state, frame_idx, camera_index, rtmp_source, camera_id, save_frame=True,
):
    """
    Motion detection function – performing all identification and marking steps according to requirements.
    """

    if state is None:
        state = {
            'history_pixels': [[deque(maxlen=REF_FRAMES) for _ in range(N_COLS)] for _ in range(N_ROWS)],
            'window_max_history': [[deque(maxlen=MAX_HISTORY_WINDOWS) for _ in range(N_COLS)] for _ in range(N_ROWS)],
            'ref_max': [[None for _ in range(N_COLS)] for _ in range(N_ROWS)],
            'ref_mean': [[None for _ in range(N_COLS)] for _ in range(N_ROWS)],
            'deviation_streak': [[deque(maxlen=TRIGGER_WINDOW_FRAMES) for _ in range(N_COLS)] for _ in range(N_ROWS)],
            'sample_rate': int(fps / SAMPLE_FPS)
        }

    trigger_regions = []
    display_frame = frame.copy()

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray_prev_frame, gray_frame)
    _, thresh = cv2.threshold(diff, THRESH_BINARY, 255, cv2.THRESH_BINARY)
    kernel = np.ones((MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    h, w = gray_frame.shape
    region_h = h // N_ROWS
    region_w = w // N_COLS

    for i in range(N_ROWS):
        for j in range(N_COLS):
            y1 = i * region_h
            y2 = (i + 1) * region_h if i < N_ROWS - 1 else h
            x1 = j * region_w
            x2 = (j + 1) * region_w if j < N_COLS - 1 else w

            region_mask = thresh[y1:y2, x1:x2]
            changed_pixels = int(np.sum(region_mask > 0))
            state['history_pixels'][i][j].append(changed_pixels)

            # Average maximum update
            if (frame_idx // state['sample_rate']) % REF_FRAMES == 0:
                if len(state['history_pixels'][i][j]) == REF_FRAMES:
                    window_max = np.max(state['history_pixels'][i][j])
                    state['window_max_history'][i][j].append(window_max)
                    if len(state['window_max_history'][i][j]) == MAX_HISTORY_WINDOWS:
                        state['ref_max'][i][j] = np.mean(state['window_max_history'][i][j])
                    else:
                        state['ref_max'][i][j] = window_max
                    state['ref_mean'][i][j] = np.mean(state['history_pixels'][i][j])

            # Calculating average color change (dominant color diff %)
            region_frame = frame[y1:y2, x1:x2]
            region_prev = prev_frame[y1:y2, x1:x2]
            mean_color_now = np.mean(region_frame.reshape(-1, 3), axis=0)
            mean_color_prev = np.mean(region_prev.reshape(-1, 3), axis=0)
            color_dist = np.linalg.norm(mean_color_now - mean_color_prev)
            color_diff_pct = 100 * color_dist / 441.7

            # Dynamic trigger conditions
            color_trigger = color_diff_pct > COLOR_CHANGE_PCT
            if state['ref_max'][i][j] is not None:
                if 0 <= state['ref_max'][i][j] <= 10:
                    trigger_threshold = state['ref_max'][i][j] + ABS_DELTA_MIN
                    is_deviation = changed_pixels > trigger_threshold
                else:
                    trigger_threshold = state['ref_max'][i][j] * (1 + DELTA_RATIO)
                    is_deviation = changed_pixels > trigger_threshold
                full_trigger = is_deviation and color_trigger
                state['deviation_streak'][i][j].append(1 if full_trigger else 0)
            else:
                state['deviation_streak'][i][j].append(0)

            is_trigger = sum(state['deviation_streak'][i][j]) >= TRIGGER_STREAK

            if is_trigger:
                trigger_regions.append((i, j, x1, y1, x2, y2))
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                state['deviation_streak'][i][j].clear()

            # Displaying a numerical value (pixels) on each square
            text = f"{changed_pixels}"
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (180,180,180), 1)
            cv2.putText(display_frame, text, (x1+5, y1+25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,200,0), 1)

    is_detected = 1 <= len(trigger_regions) <= 2

    summary = "YES - change" if is_detected else "NO - no change"
    cv2.rectangle(display_frame, (10, 10), (370, 50), (0,0,0), -1)
    cv2.putText(display_frame, summary, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0) if is_detected else (0,0,255), 2)

    print(f"{camera_index} object_96 frame count {frame_idx}: detected motion: {is_detected}")
    # Save image with markup (if desired)
    if is_detected and save_frame:
        timestamp = int(datetime.datetime.utcnow().timestamp()*1000)
        fname = os.path.join(OUTPUT_FOLDER, f"motion_{rtmp_source}_{camera_id}_{timestamp}.jpg")
        cv2.imwrite(fname, display_frame)
        print(f"{camera_index} object_96 motion frame saved to: {fname}")

    return is_detected, state

def ceil_dt(dt, delta):
    return dt + (datetime.datetime.min - dt) % delta

def main():

    rtmp_source = os.environ.get("RTMP_SOURCE")
    camera_id = os.environ.get("CAMERA_ID")
    print("Starting stream_view from main()")
    stream_view(rtmp_source, camera_id)

if __name__ == "__main__":
    main()

