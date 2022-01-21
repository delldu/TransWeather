import image_weather

image_weather.image_client("TAI", "images/input/*.png", "output")
image_weather.image_server("TAI")

image_weather.video_client("TAI", "/home/dell/noise.mp4", "output/server_clean.mp4")
image_weather.video_server("TAI")

image_weather.image_predict("images/input/*.png", "output")
image_weather.video_predict("/home/dell/noise.mp4", "output/predict_clean.mp4")

