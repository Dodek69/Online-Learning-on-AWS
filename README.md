# **Asynchronous Learning with Kafka**

---

**Dominik Ciołczyk, Urszula Kicinger, Ewa Żukowska**

---

## Overview

The objective of this project was to build an asynchronous architecture that uses Apache Kafka for message streaming, enabling automated image processing and machine learning model training in a streaming fashion. The system includes a producer to send images to a Kafka topic and a consumer to process the images, train the model, and monitor its accuracy.

---

## Technologies Used

- **Apache Kafka:** Interface to interact with Apache Kafka.
- **Python:** Core programming language.
- **Docker:** For containerizing Kafka and Zookeeper services.
- **Amazon EC2:** Hosting the project in a cloud environment.

---

## System Architecture

### **Producer**
- Monitors a local folder (`./images`) for new images.
- Processes images (resizing to `224×224` pixels, converting to byte format).
- Sends images as messages to Kafka.

### **Kafka Broker**
- Handles message transmission between the producer and the consumer.
- Configured using Docker Compose with Zookeeper as a back-end.

### **Consumer**
- Receives images from Kafka.
- Processes the images and uses them to train a machine learning model in real-time.
- Monitors the model's accuracy during training.

### **Machine Learning Model**
- Neural network architecture built using PyTorch.
- Model created with `river`: Classifier `deep-river` with a `binary_cross_entropy` loss function and `sgd` optimizer.

---
