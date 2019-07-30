from crontab import CronTab

cron= CronTab(user='nvidia')
job=cron.new(command='python ~/Desktop/People-counter-using-Dlib-facial-recognition-and-tracking/face_recognizer.py')
job.minute.every(1)

for item in cron:
    print(item)

cron.write()

