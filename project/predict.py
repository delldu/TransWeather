import clean_weather

# clean_weather.image_client("PAI", "images/input/*.png", "output")
# clean_weather.image_server("PAI")

# clean_weather.video_client("PAI", "/home/dell/noise.mp4", "/tmp/output.mp4")
# clean_weather.video_server("PAI")

clean_weather.image_predict("images/input/*.png", "output")
# clean_weather.video_predict("/home/dell/noise.mp4", "/tmp/clean.mp4")

