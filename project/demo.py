import image_derain

# image_derain.image_client("TAI", "images/input/*.png", "output")
# image_derain.image_server("TAI")

# image_derain.video_client("TAI", "/home/dell/noise.mp4", "output/server_clean.mp4")
# image_derain.video_server("TAI")

# image_derain.image_predict("images/input/*.png", "output")
# image_derain.video_predict("/home/dell/noise.mp4", "output/predict_clean.mp4")

image_derain.image_predict("images/*.png", "output")
