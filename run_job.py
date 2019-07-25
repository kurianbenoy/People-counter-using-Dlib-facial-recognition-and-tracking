from crontab import CronTab

cron= CronTab()
job=cron.new(command='print_o.py')

