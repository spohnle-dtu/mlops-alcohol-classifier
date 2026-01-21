from locust import HttpUser, task, between

class BeverageApiUser(HttpUser):
    # Simulate a user waiting between 1 and 2 seconds between requests
    wait_time = between(1, 2)

    @task
    def health_check(self):
        """Simulate a user checking the API status."""
        self.client.get("/health")

    @task(3) # This task happens 3x more often than the health check
    def predict_request(self):
        """Simulate a user uploading an image for prediction."""
        # Use a small dummy file or a real sample from your data directory
        files = {"image": ("test.jpg", b"fake_data", "image/jpeg")}
        self.client.post("/predict", files=files)
