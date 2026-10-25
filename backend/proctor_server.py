from flask import Flask, request
from flask_socketio import SocketIO, emit

import base64
import cv2
import numpy as np
import time
import sys
import threading
import gc
import os

from pathlib import Path


# ============================================================
# IMPORT
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

sys.path.insert(
    0,
    str(BASE_DIR)
)

from ai_proctor import InterviewObjectDetector


# ============================================================
# FLASK
# ============================================================

app = Flask(__name__)

app.config["SECRET_KEY"] = (
    "ai-interview-proctoring"
)


socketio = SocketIO(

    app,

    cors_allowed_origins="*",

    async_mode="threading",

    max_http_buffer_size=4_000_000,

    ping_timeout=60,

    ping_interval=25
)


# ============================================================
# DETECTORS
# ============================================================

detectors = {}

detector_locks = {}

detectors_lock = threading.Lock()


# ============================================================
# GET DETECTOR
# ============================================================

def get_detector(sid):

    with detectors_lock:

        if sid not in detectors:

            print()
            print(
                "Creating detector for:",
                sid
            )

            detectors[sid] = (
                InterviewObjectDetector(
                    use_camera=False
                )
            )

            detector_locks[sid] = (
                threading.Lock()
            )

        return (
            detectors[sid],
            detector_locks[sid]
        )


# ============================================================
# SERIALIZE RESULT
# ============================================================

def serialise_result(
    detector,
    result
):

    return {

        # ----------------------------------------------------
        # FACE
        # ----------------------------------------------------

        "face_detected":
            bool(
                result.get(
                    "face_detected",
                    False
                )
            ),

        "face_inside_circle":
            bool(
                result.get(
                    "face_inside_circle",
                    False
                )
            ),

        "face_misaligned":
            bool(
                result.get(
                    "face_misaligned",
                    False
                )
            ),

        "face_reason":
            result.get(
                "face_reason",
                ""
            ),

        "face_box":
            result.get(
                "face_box"
            ),

        "face_center":
            result.get(
                "face_center"
            ),

        # ----------------------------------------------------
        # PEOPLE
        # ----------------------------------------------------

        "multiple_people":
            bool(
                result.get(
                    "multiple_people",
                    False
                )
            ),

        "people_count":
            int(
                result.get(
                    "people_count",
                    0
                )
            ),

        "people":
            result.get(
                "people",
                []
            ),

        # ----------------------------------------------------
        # PHONE
        # ----------------------------------------------------

        "cell_phone":
            bool(
                result.get(
                    "cell_phone",
                    False
                )
            ),

        "phones":
            result.get(
                "phones",
                []
            ),

        # ----------------------------------------------------
        # OBJECTS
        # ----------------------------------------------------

        "objects":
            result.get(
                "objects",
                []
            ),

        "hand_detected":
            bool(
                result.get(
                    "hand_detected",
                    False
                )
            ),

        "hand_carrying_object":
            bool(
                result.get(
                    "hand_carrying_object",
                    False
                )
            ),

        "carried_object":
            result.get(
                "carried_object"
            ),

        # ----------------------------------------------------
        # LOOKING
        # ----------------------------------------------------

        "looking_direction":
            result.get(
                "looking_direction",
                "CENTER"
            ),

        "yaw":
            float(
                result.get(
                    "yaw",
                    0.0
                )
            ),

        "pitch":
            float(
                result.get(
                    "pitch",
                    0.0
                )
            ),

        "roll":
            float(
                result.get(
                    "roll",
                    0.0
                )
            ),

        "side_duration":
            float(
                result.get(
                    "side_duration",
                    0.0
                )
            ),

        # ----------------------------------------------------
        # FACE TIMERS
        # ----------------------------------------------------

        "no_face":
            bool(
                result.get(
                    "no_face",
                    False
                )
            ),

        "no_face_duration":
            float(
                result.get(
                    "no_face_duration",
                    0.0
                )
            ),

        "misaligned_duration":
            float(
                result.get(
                    "misaligned_duration",
                    0.0
                )
            ),

        # ----------------------------------------------------
        # WARNINGS
        # ----------------------------------------------------

        "warning_count":
            int(
                result.get(
                    "warning_count",
                    detector.warning_count
                )
            ),

        "max_warnings":
            5,

        "last_warning_reason":
            result.get(
                "last_warning_reason",
                detector.last_warning_reason
            ),

        "warnings":
            result.get(
                "warnings",
                []
            ),

        # ----------------------------------------------------
        # TERMINATION
        # ----------------------------------------------------

        "terminated":
            bool(
                result.get(
                    "terminated",
                    False
                )
            )
    }


# ============================================================
# CONNECT
# ============================================================

@socketio.on("connect")
def connected():

    sid = request.sid

    print()
    print(
        "Browser connected:",
        sid
    )

    # IMPORTANT:
    #
    # Do NOT create YOLO here.
    #
    # Detector is created only when
    # the first video frame arrives.

    emit(
        "proctor_status",
        {
            "connected": True
        }
    )


# ============================================================
# DISCONNECT
# ============================================================

@socketio.on("disconnect")
def disconnected():

    sid = request.sid

    print()
    print(
        "Browser disconnected:",
        sid
    )

    with detectors_lock:

        detector = detectors.pop(
            sid,
            None
        )

        detector_locks.pop(
            sid,
            None
        )

    if detector:

        try:

            detector.cleanup()

        except Exception as e:

            print(
                "Detector cleanup error:",
                e
            )

    gc.collect()


# ============================================================
# RESET
# ============================================================

@socketio.on("reset_proctoring")
def reset_proctoring():

    sid = request.sid

    detector, lock = get_detector(
        sid
    )

    with lock:

        detector.reset_warnings()

    emit(
        "proctor_result",
        {
            "warning_count": 0,
            "terminated": False
        }
    )


# ============================================================
# VIDEO FRAME
# ============================================================

@socketio.on("video_frame")
def video_frame(payload):

    sid = request.sid

    try:

        # ----------------------------------------------------
        # GET DETECTOR
        # ----------------------------------------------------

        detector, lock = get_detector(
            sid
        )

        # ----------------------------------------------------
        # GET IMAGE
        # ----------------------------------------------------

        if isinstance(
            payload,
            dict
        ):

            data = payload.get(
                "image",
                ""
            )

        else:

            data = str(payload)

        if not data:

            emit(
                "proctor_error",
                {
                    "error":
                        "No image received"
                }
            )

            return

        # ----------------------------------------------------
        # REMOVE DATA URL
        # ----------------------------------------------------

        if "," in data:

            data = data.split(
                ",",
                1
            )[1]

        # ----------------------------------------------------
        # DECODE
        # ----------------------------------------------------

        raw = base64.b64decode(
            data
        )

        frame = cv2.imdecode(
            np.frombuffer(
                raw,
                dtype=np.uint8
            ),
            cv2.IMREAD_COLOR
        )

        if frame is None:

            emit(
                "proctor_error",
                {
                    "error":
                        "Could not decode webcam frame"
                }
            )

            return

        # ----------------------------------------------------
        # SERVER-SIDE MONOTONIC TIMESTAMP
        # ----------------------------------------------------

        timestamp_ms = int(
            time.monotonic() * 1000
        )

        # ----------------------------------------------------
        # CRITICAL LOCK
        #
        # Only ONE frame from this browser
        # enters the detector at a time.
        # ----------------------------------------------------

        with lock:

            result = (
                detector.process_frame(
                    frame,
                    timestamp_ms
                )
            )

        # ----------------------------------------------------
        # SERIALIZE
        # ----------------------------------------------------

        response = serialise_result(
            detector,
            result
        )

        # ----------------------------------------------------
        # SEND TO FRONTEND
        # ----------------------------------------------------

        emit(
            "proctor_result",
            response
        )

    except Exception as e:

        print()
        print(
            "Frame processing error:",
            repr(e)
        )

        emit(
            "proctor_error",
            {
                "error":
                    str(e)
            }
        )


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    return {
        "ok": True,
        "service":
            "AI interview proctoring"
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print()
    print(
        "AI Interview Proctoring WebSocket "
        "server running on "
        "http://localhost:5000"
    )

    socketio.run(

        app,

        host="0.0.0.0",

        port=int(os.environ.get('PROCTOR_PORT','5000')),

        debug=False,

        allow_unsafe_werkzeug=True
    )