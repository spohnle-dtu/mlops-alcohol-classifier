from locust import HttpUser, between, task


class BeverageApiUser(HttpUser):
    # Simulate a user waiting between 1 and 2 sec
    wait_time = between(1, 2)

    @task
    def health_check(self):
        """Simulate a user checking the API status."""
        self.client.get("/health")

    @task(3)
    def predict_request(self):
        """Simulate a user uploading an image for prediction."""

        files = {"image": ("test.jpg", b"fake_data", "image/jpeg")}
        self.client.post("/predict", files=files)
