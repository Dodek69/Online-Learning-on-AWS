import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from kafka import KafkaProducer

# Kafka producer setup
producer = KafkaProducer(bootstrap_servers='localhost:9092')

# Folder to monitor where the pathologists will dump their images
folder_to_watch = '...'

# Event handler for new files
class NewFileHandler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory:
            file_path = event.src_path
            send_image_to_kafka(file_path)

# Function to send image to Kafka
def send_image_to_kafka(file_path):
    with open(file_path, 'rb') as f:
        image_data = f.read()
        producer.send('image_topic', value=image_data)
        print(f"Produced image {file_path} and sent to Kafka")

# Start monitoring folder for new files
observer = Observer()
observer.schedule(NewFileHandler(), path=folder_to_watch, recursive=False)
observer.start()

try:
    while True:
        time.sleep(1)  # Keep the observer thread running
except KeyboardInterrupt:
    observer.stop()

observer.join()

# Close Kafka producer
producer.close()

