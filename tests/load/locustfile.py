from locust import HttpUser, task, between
import os

class WebsiteUser(HttpUser):
    wait_time = between(1, 5)

    @task
    def detect_image(self):
        # Path to the test image
        test_image_path = "data/test/real/real_002.jpg"
        
        if not os.path.exists(test_image_path):
            print(f"Error: Test image not found at {test_image_path}")
            return

        with open(test_image_path, "rb") as image:
            files = {"file": ("real_002.jpg", image, "image/jpeg")}
            self.client.post("/api/detect", files=files)
