# Exam template for 02476 Machine Learning Operations

This is the report template for the exam. Please only remove the text formatted as with three dashes in front and behind
like:

```--- question 1 fill here ---```

Where you instead should add your answers. Any other changes may have unwanted consequences when your report is
auto-generated at the end of the course. For questions where you are asked to include images, start by adding the image
to the `figures` subfolder (please only use `.png`, `.jpg` or `.jpeg`) and then add the following code in your answer:

`![my_image](figures/<image>.<extension>)`

In addition to this markdown file, we also provide the `report.py` script that provides two utility functions:

Running:

```bash
python report.py html
```

Will generate a `.html` page of your report. After the deadline for answering this template, we will auto-scrape
everything in this `reports` folder and then use this utility to generate a `.html` page that will be your serve
as your final hand-in.

Running

```bash
python report.py check
```

Will check your answers in this template against the constraints listed for each question e.g. is your answer too
short, too long, or have you included an image when asked. For both functions to work you mustn't rename anything.
The script has two dependencies that can be installed with

```bash
pip install typer markdown
```

or

```bash
uv add typer markdown
```

## Overall project checklist

The checklist is *exhaustive* which means that it includes everything that you could do on the project included in the
curriculum in this course. Therefore, we do not expect at all that you have checked all boxes at the end of the project.
The parenthesis at the end indicates what module the bullet point is related to. Please be honest in your answers, we
will check the repositories and the code to verify your answers.

### Week 1

* [x] Create a git repository (M5)
* [x] Make sure that all team members have write access to the GitHub repository (M5)
* [x] Create a dedicated environment for you project to keep track of your packages (M2)
* [x] Create the initial file structure using cookiecutter with an appropriate template (M6)
* [x] Fill out the `data.py` file such that it downloads whatever data you need and preprocesses it (if necessary) (M6)
* [x] Add a model to `model.py` and a training procedure to `train.py` and get that running (M6)
* [x] Remember to either fill out the `requirements.txt`/`requirements_dev.txt` files or keeping your
    `pyproject.toml`/`uv.lock` up-to-date with whatever dependencies that you are using (M2+M6)
* [x] Remember to comply with good coding practices (`pep8`) while doing the project (M7)
* [x] Do a bit of code typing and remember to document essential parts of your code (M7)
* [x] Setup version control for your data or part of your data (M8)
* [x] Add command line interfaces and project commands to your code where it makes sense (M9)
* [x] Construct one or multiple docker files for your code (M10)
* [x] Build the docker files locally and make sure they work as intended (M10)
* [x] Write one or multiple configurations files for your experiments (M11)
* [x] Used Hydra to load the configurations and manage your hyperparameters (M11)
* [ ] Use profiling to optimize your code (M12)
* [x] Use logging to log important events in your code (M14)
* [x] Use Weights & Biases to log training progress and other important metrics/artifacts in your code (M14)
* [x] Consider running a hyperparameter optimization sweep (M14)
* [ ] Use PyTorch-lightning (if applicable) to reduce the amount of boilerplate in your code (M15)

### Week 2

* [x] Write unit tests related to the data part of your code (M16)
* [x] Write unit tests related to model construction and or model training (M16)
* [x] Calculate the code coverage (M16)
* [x] Get some continuous integration running on the GitHub repository (M17)
* [x] Add caching and multi-os/python/pytorch testing to your continuous integration (M17)
* [x] Add a linting step to your continuous integration (M17)
* [x] Add pre-commit hooks to your version control setup (M18)
* [x] Add a continues workflow that triggers when data changes (M19)
* [x] Add a continues workflow that triggers when changes to the model registry is made (M19)
* [x] Create a data storage in GCP Bucket for your data and link this with your data version control setup (M21)
* [x] Create a trigger workflow for automatically building your docker images (M21)
* [x] Get your model training in GCP using either the Engine or Vertex AI (M21)
* [x] Create a FastAPI application that can do inference using your model (M22)
* [x] Deploy your model in GCP using either Functions or Run as the backend (M23)
* [x] Write API tests for your application and setup continues integration for these (M24)
* [ ] Load test your application (M24)
* [ ] Create a more specialized ML-deployment API using either ONNX or BentoML, or both (M25)
* [x] Create a frontend for your API (M26)

### Week 3

* [ ] Check how robust your model is towards data drifting (M27)
* [x] Setup collection of input-output data from your deployed application (M27)
* [ ] Deploy to the cloud a drift detection API (M27)
* [ ] Instrument your API with a couple of system metrics (M28)
* [ ] Setup cloud monitoring of your instrumented application (M28)
* [ ] Create one or more alert systems in GCP to alert you if your app is not behaving correctly (M28)
* [ ] If applicable, optimize the performance of your data loading using distributed data loading (M29)
* [ ] If applicable, optimize the performance of your training pipeline by using distributed training (M30)
* [ ] Play around with quantization, compilation and pruning for you trained models to increase inference speed (M31)

### Extra

* [x] Write some documentation for your application (M32)
* [ ] Publish the documentation to GitHub Pages (M32)
* [x] Revisit your initial project description. Did the project turn out as you wanted?
* [x] Create an architectural diagram over your MLOps pipeline
* [x] Make sure all group members have an understanding about all parts of the project
* [x] Uploaded all your code to GitHub

## Group information

### Question 1
> **Enter the group number you signed up on <learn.inside.dtu.dk>**
>
> Answer:

149

### Question 2
> **Enter the study number for each member in the group**
>
> Example:
>
> *sXXXXXX, sXXXXXX, sXXXXXX*
>
> Answer:

s204151, s242875, s242708, s242845, s174035(ralor)

### Question 3
> **Did you end up using any open-source frameworks/packages not covered in the course during your project? If so**
> **which did you use and how did they help you complete the project?**
>
> Recommended answer length: 0-200 words.
>
> Example:
> *We used the third-party framework ... in our project. We used functionality ... and functionality ... from the*
> *package to do ... and ... in our project*.
>
> Answer:

We used pre-trained ResNet18 for the computer-vision part of the project and then used transfer-learning to make it fit a 3-class classification problem.

## Coding environment

> In the following section we are interested in learning more about you local development environment. This includes
> how you managed dependencies, the structure of your code and how you managed code quality.

### Question 4

> **Explain how you managed dependencies in your project? Explain the process a new team member would have to go**
> **through to get an exact copy of your environment.**
>
> Recommended answer length: 100-200 words
>
> Example:
> *We used ... for managing our dependencies. The list of dependencies was auto-generated using ... . To get a*
> *complete copy of our development environment, one would have to run the following commands*
>
> Answer:

We managed dependencies using standard Python pip requirments files. To run the core runtime dependencies for training, evaluation, and the FastAPI backend were placed in requirements.txt, while data/versioning dependencies were placed into requirements_dvc.txt (for DVC and the chosen remote storage). This split keeps the base environment lightweight while still supporting reproducible data and model retrieval.
To recreate the exact environment, a new team member clones the repository, creates an isolated virtual environment, installs dependencies from both requirements files, and then pulls the versioned data/models with DVC. In addition, since we log experiments to Weights & Biases, the member must be invited to our W&B entity/project and authenticate locally before running training.

An example would be:

git clone <repo-url>
pip install -r requirements.txt
pip install -r requirements_dvc.txt
dvc pull
wandb login



### Question 5

> **We expect that you initialized your project using the cookiecutter template. Explain the overall structure of your**
> **code. What did you fill out? Did you deviate from the template in some way?**
>
> Recommended answer length: 100-200 words
>
> Example:
> *From the cookiecutter template we have filled out the ... , ... and ... folder. We have removed the ... folder*
> *because we did not use any ... in our project. We have added an ... folder that contains ... for running our*
> *experiments.*
>
> Answer:
> We started out using the Cookiecutter tempate given by: https://github.com/SkafteNicki/mlops_template, and mostly adhered to that structure. The src/ directory was filled with the core runtime code, including data preprocessing, dataset handling, model definition, training, evaluation, and a FastAPI-based inference API. This is where most of the project-specific logic lives. We made small test scripts in test folder. We added small, focused test scripts in the tests/ folder to validate key components such as data loading, model behavior, and the API. Configuration files are stored in the configs/ directory and managed with Hydra, as intended by the template.

In addition to the original structure, we introduced dedicated dockerfiles/ and frontend/ folders to support containerized deployment and a simple Streamlit-based frontend. Finally, the checkpoints/ directory is tracked with DVC and used to store the best-performing trained model.

### Question 6

> **Did you implement any rules for code quality and format? What about typing and documentation? Additionally,**
> **explain with your own words why these concepts matters in larger projects.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We used ... for linting and ... for formatting. We also used ... for typing and ... for documentation. These*
> *concepts are important in larger projects because ... . For example, typing ...*
>
> Answer:

We implemented several rules and tools to ensure code quality and readability throughout the project. We used Ruff as our primary tool for both linting and formatting. Ruff ensures a consistent coding style and detects if there are any unused imports, undefined variables, and other issues which help catching bugs early on in the development and reduces redundancy.

Additionally, we made use of Python type hints in function signatures and class definitions to make it easier to understand the inputs required, and the expected outputs. Type annotations are used in the core parts of the pipeline (data loading, model interfaces, and training loops), making the code easier to reason about and less error-prone, but it is not necessairly everywhere. The same applies for inline comments that we used to describe parts of the code.

These kind of practices and tools are crucial in bigger projects where multiple developers work together. Consistent formatting, linting, and tracking unsued parts of the code, ensures structure and readability of code. It also makes it easier for co-developers to read and understand changes. In general, these practices lowers the barrier for new contributors, and easies the integration of new code.

## Version control

> In the following section we are interested in how version control was used in your project during development to
> corporate and increase the quality of your code.

### Question 7

> **How many tests did you implement and what are they testing in your code?**
>
> Recommended answer length: 50-100 words.
>
> Example:
> *In total we have implemented X tests. Primarily we are testing ... and ... as these the most critical parts of our*
> *application but also ... .*
>
> Answer:

We implemented three test modules (test_data.py, test_model.py, and test_api.py) which were mainly used in the begginning og the project. For FastAPI, the API tests validate the /health endpoint to ensure the service is running correctly, and the /predict endpoint to confirm it accepts image uploads and returns a valid prediction response with class labels and probabilities.In addition, we used Weights & Biases (W&B) to log training and evaluation metrics, which helps identify regressions and monitor the impact of code and data changes over time. This makes 14 Unit tests in total


### Question 8

> **What is the total code coverage (in percentage) of your code? If your code had a code coverage of 100% (or close**
> **to), would you still trust it to be error free? Explain you reasoning.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *The total code coverage of code is X%, which includes all our source code. We are far from 100% coverage of our **
> *code and even if we were then...*
>
> Answer:

Our total code coverage is 65%. Coverage can be used as an indicator that our core modules (data.py, model.py, train.py etc) are exercised automatically, and that changes that we might do will not silently break the functionality or core modules.

We would never trust a code to be completely error-free even though the coverage is 100%. A coverage of 100% tells us that all our lines were executed, but not necessarily that they are tested with the right edge cases or guarentee that the logic in our code was validated in a meaningfull way.

Doing an API test and calling the /predict endpoint with a valid response, might still overlook problems with mismatching of class labels, or preporcessing which is inconsistent etc. Moreover, training code might execute error-free while still hiding problems with correct metrics or reproducebility. Thus, reliability also depends on well-designed tests, continous monitoring and loggin with wandb for instance, and evaluation of real data.

### Question 9

> **Did you workflow include using branches and pull requests? If yes, explain how. If not, explain how branches and**
> **pull request can help improve version control.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We made use of both branches and PRs in our project. In our group, each member had an branch that they worked on in*
> *addition to the main branch. To merge code we ...*
>
> Answer:

We made use of branches in this project, where each group member had their own branch to work on new functionality without affecting the main codebase. Before committing new code, the latest version of the main branch was first pulled and merged into the member’s branch to stay up to date and resolve any potential conflicts early. Once the changes were tested and working as expected, a clean merge was performed back into the main branch.

We did not use PRs in this project. However, PRs could have been used to allow other group members to review the code before merging, which would make it easier to catch bugs, suggest improvements, and maintain a higher overall code quality.

### Question 10

> **Did you use DVC for managing data in your project? If yes, then how did it improve your project to have version**
> **control of your data. If no, explain a case where it would be beneficial to have version control of your data.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We did make use of DVC in the following way: ... . In the end it helped us in ... for controlling ... part of our*
> *pipeline*
>
> Answer:

We used DVC to manage data and model artifacts in the project, e.g. DVC was used to track the raw data ona Google Drive folder, instead of committing large datasets and trained models directly to Git. This helped keep the repository clean and lightweight while still allowing us to version control important data outputs such as processed datasets and the best model checkpoint.

Using DVC made our collaboration efforts easier, since everyone on the team could pull the same data and model versions with dvc pull, reducing confusion about which data was used for training and evaluation. It also helped with reproducibility, as each Git commit is linked to a specific data state. It initially added some extra setup and commands, but overall DVC improved the structure and reliability of our ML workflow.

### Question 11

> **Discuss you continuous integration setup. What kind of continuous integration are you running (unittesting,**
> **linting, etc.)? Do you test multiple operating systems, Python  version etc. Do you make use of caching? Feel free**
> **to insert a link to one of your GitHub actions workflow.**
>
> Recommended answer length: 200-300 words.
>
> Example:
> *We have organized our continuous integration into 3 separate files: one for doing ..., one for running ... testing*
> *and one for running ... . In particular for our ..., we used ... .An example of a triggered workflow can be seen*
> *here: <weblink>*
>
> Answer:


Our continuous integration (CI) setup is consists of various GitHub Actions workflows to ensure code quality and reliability. We have workflows for unit testing, and building Docker images. These worksflows are executes every time a change is committed to the main branch. Python version 3.12 was tested in all workflows and Ubuntu was tested in all workflows besides tests.yml which runs the latest version of Windows.

File `.github/workflows/tests.yml` runs Ruff to check for code style/formatting issues and uses pytest to run all unit tests.

File `.github/workflows/docker-build.yml` builds Docker images. This quality checks our Dockerfiles and checks if they can be built successfully.

File `.github/workflows/data-changes.yml` does data testing (runs `tests/test_data.py`) to ensure that data changes don't break anything.

File `.github/workflows/model-registry-changes.yml` tests the model training/creation and tests linting. It runs `tests/tests/test_model.py` and `tests/test_train_step.py`

An example of a triggered workflow can be seen here: https://github.com/spohnle-dtu/mlops-alcohol-classifier/actions/runs/21292282158


## Running code and tracking experiments

> In the following section we are interested in learning more about the experimental setup for running your code and
> especially the reproducibility of your experiments.

### Question 12

> **How did you configure experiments? Did you make use of config files? Explain with coding examples of how you would**
> **run a experiment.**
>
> Recommended answer length: 50-100 words.
>
> Example:
> *We used a simple argparser, that worked in the following way: Python  my_script.py --lr 1e-3 --batch_size 25*
>
> Answer:

We configured experiments using Hydra config files stored in the configs/ folder (e.g., dataset paths, batch size, learning rate, epochs, and model settings). This made it easy to reproduce runs and override parameters from the command line without changing code. For example, a default training run is started with:

python -m src.alcohol_classifier.train

and a modified experiment can be launched by overriding config values:

python -m src.alcohol_classifier.train model.lr=1e-3 model.epochs=20 dataset.batch_size=128


### Question 13

> **Reproducibility of experiments are important. Related to the last question, how did you secure that no information**
> **is lost when running experiments and that your experiments are reproducible?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We made use of config files. Whenever an experiment is run the following happens: ... . To reproduce an experiment*
> *one would have to do ...*
>
> Answer:

We ensured reproducibility by making use of config management, data versioning, and experiment logging. 

Our experiment configurations are defined through Hydra config files, so hyperparameters, dataset paths, and model settings are explicitly specified and not hard-coded. For datapaths, we made sure to always define paths from the root folder. When an experiment is run, Hydra automatically saves the fully resolved configuration to the output directory, ensuring that the exact setup of each run is preserved.

For version control we used DVC for data and model checkpoints, which allows us to tie experiment with a specific data and model state. In addition, training and evaluation metrics are logged using Weights & Biases (W&B), providing a persistent record of results and metadata. 

### Question 14

> **Upload 1 to 3 screenshots that show the experiments that you have done in W&B (or another experiment tracking**
> **service of your choice). This may include loss graphs, logged images, hyperparameter sweeps etc. You can take**
> **inspiration from [this figure](figures/wandb.png). Explain what metrics you are tracking and why they are**
> **important.**
>
> Recommended answer length: 200-300 words + 1 to 3 screenshots.
>
> Example:
> *As seen in the first image when have tracked ... and ... which both inform us about ... in our experiments.*
> *As seen in the second image we are also tracking ... and ...*
>
> Answer:

![wandb_1](figures/wandb_1.png)
![wandb_3](figures/wandb_3.png)
![wandb_2](figures/wandb_2.png)

The screenshots above show experiments that were tracked using Weights & Biases (W&B) while training and validating our alcohol classification model.
In the first image, we track training/validation loss and accuracy across epochs. The training loss drops quickly and the accuracy increases to almost 100%, which shows that the model is learning the training data effectively and that the optimization setup works as intended. However, even though the validation shows similar trends, it is much more variable and changes accross epochs, which is a sign of overfitting.

The second image shows validation loss and validation accuracy accross different runs. Here we can observe significant differences between some runs, and we can investigate the model setup for each indivual run to get closer to what actually works in our models, and what does not.


In the third image, we log system-level metrics, such as GPU clock speed during training. These metrics are useful for understanding hardware performance, comparing runs on different machines, and explaining differences in training time. They also help with debugging performance issues when running experiments on GPUs.

Overall, W&B allows us to keep track of metrics, hyperparameters, and experiment history in one place. This makes it easier to compare runs, reproduce results, and understand how different configuration choices affect both model performance and training efficiency.

### Question 15

> **Docker is an important tool for creating containerized applications. Explain how you used docker in your**
> **experiments/project? Include how you would run your docker images and include a link to one of your docker files.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *For our project we developed several images: one for training, inference and deployment. For example to run the*
> *training docker image: `docker run trainer:latest lr=1e-3 batch_size=64`. Link to docker file: <weblink>*
>
> Answer:

********* WIRTE SOMETHING HERE ***********

### Question 16

> **When running into bugs while trying to run your experiments, how did you perform debugging? Additionally, did you**
> **try to profile your code or do you think it is already perfect?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *Debugging method was dependent on group member. Some just used ... and others used ... . We did a single profiling*
> *run of our main code at some point that showed ...*
>
> Answer:

During development we mainly ran into bugs related to data loading, config mismatches, and differences between local runs and Docker/Cloud setups. We debugged these issues step by step using simple logging, print statements, and unit tests. When something broke, we often ran individual modules like data.py, train.py, or evaluate.py on their own and added checks for tensor shapes, data types, and file paths before running the full pipeline. Pytest was used to test core pieces such as a single training step, which helped catch errors early.
Several problems were also caused by the runtime environment, for example missing files inside Docker images or wrong paths. These were debugged by running containers interactively and inspecting files and environment variables.

We did not perform extensive profiling, as the model and dataset size were relatively small and not limited by computing power. However, we did monitored training time per epoch and GPU/CPU utilization to ensure there were no obvious bottlenecks, but also as a mean of debugging.

## Working in the cloud

> In the following section we would like to know more about your experience when developing in the cloud.

### Question 17

> **List all the GCP services that you made use of in your project and shortly explain what each service does?**
>
> Recommended answer length: 50-200 words.
>
> Example:
> *We used the following two services: Engine and Bucket. Engine is used for... and Bucket is used for...*
>
> Answer:

In this project we used several Google Cloud Platform (GCP) services. We aimed at using it as an integrated part of our project, but we had quite a few issues with creating fast access for each team member, so it did not become as integrated as we aimed @. However, we did do the following:


A Google Cloud Storage Bucket was used to store raw data and trained model artifacts. The bucket was also used together with DVC for data and model versioning, making experiments reproducible across team members.

We used Google Cloud Run was used to deploy the trained model as a containerized FastAPI service. That allowed us to run Docker images without managing servers.

Google Cloud Build was used to build Docker images from the GitHub repository and push them to the registry as part of deployment.

Finally, IAM (Identity and Access Management) was used to control access, for example granting collaborators permissions to the storage bucket, but several team members had problems in this step, and we did not manage to integrate a fast access possibility locally.

### Question 18

> **The backbone of GCP is the Compute engine. Explained how you made use of this service and what type of VMs**
> **you used?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We used the compute engine to run our ... . We used instances with the following hardware: ... and we started the*
> *using a custom container: ...*
>
> Answer:

We used Google Compute Engine to run virtual machines where we did development and testing of our project in the cloud. The VMs were used to run Docker containers for training, evaluation, and debugging, which made it easier to reproduce issues that did not appear on local machines. This helped us overcome problems with individual machines, especially RAM-related issues.
Using GCE was useful for testing access to Google Cloud Storage and checking that our containers worked correctly in a cloud environment. This allowed us to verify that the full pipeline worked outside local machines before deployment.

We used general-purpose CPU-based virtual machines, which were sufficient given the moderate size of our dataset and model. The virtual machines were mainly used to pull data from Google Cloud Storage, run training and evaluation scripts, and test Docker images in a cloud environment. 


### Question 19

> **Insert 1-2 images of your GCP bucket, such that we can see what data you have stored in it.**
> **You can take inspiration from [this figure](figures/bucket.png).**
>
> Answer:

![GCP_Bucket](figures/GCP_Bucket.png)

### Question 20

> **Upload 1-2 images of your GCP artifact registry, such that we can see the different docker images that you have**
> **stored. You can take inspiration from [this figure](figures/registry.png).**
>
> Answer:

![GCP_Docker_v1](figures/GCP_Docker_v1.png)

### Question 21

> **Upload 1-2 images of your GCP cloud build history, so we can see the history of the images that have been build in**
> **your project. You can take inspiration from [this figure](figures/build.png).**
>
> Answer:

![GCP_cloudbuild_v1](figures/GCP_cloudbuild_v1.png)

### Question 22

> **Did you manage to train your model in the cloud using either the Engine or Vertex AI? If yes, explain how you did**
> **it. If not, describe why.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We managed to train our model in the cloud using the Engine. We did this by ... . The reason we choose the Engine*
> *was because ...*
>
> Answer:

We did not fully rely on the cloud for large-scale model training. Most of the model development and training was performed locally, as the dataset and model size were relatively small and did not require significant computational resources. However, we did experiment with training and running parts of the pipeline on Google Compute Engine virtual machines to understand how the workflow would behave in a cloud environment.
On the virtual machine, we tested pulling data from Google Cloud Storage, running training and evaluation scripts, and verifying that the setup worked end-to-end outside our local machines. This helped us debug cloud-specific issues such as permissions, paths, and environment configuration. 


## Deployment

### Question 23

> **Did you manage to write an API for your model? If yes, explain how you did it and if you did anything special. If**
> **not, explain how you would do it.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We did manage to write an API for our model. We used FastAPI to do this. We did this by ... . We also added ...*
> *to the API to make it more ...*
>
> Answer:
> 
We built an API for our trained model using FastAPI. The idea was to make the model accessible as a small web service that can run predictions. The PyTorch model is loaded once when the API starts from the saved checkpoint (best.pt), so it doesn’t need to be reloaded for every request, which keeps things faster.
The API has two main endpoints. The /health endpoint is a simple check to see if the service is up and whether the model loaded correctly. This is mainly useful when deploying the service, for example on Cloud Run, to make sure everything is running as expected. The /predict endpoint is where the actual inference happens. It takes an uploaded image, applies the same preprocessing as during training (resizing and normalization), and returns the predicted class together with the model’s confidence for each class.

### Question 24

> **Did you manage to deploy your API, either in locally or cloud? If not, describe why. If yes, describe how and**
> **preferably how you invoke your deployed service?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *For deployment we wrapped our model into application using ... . We first tried locally serving the model, which*
> *worked. Afterwards we deployed it in the cloud, using ... . To invoke the service an user would call*
> *`curl -X POST -F "file=@file.json"<weburl>`*
>
> Answer:

Yes, we deployed our API both locally and in the cloud. We started by running the FastAPI service locally to check that everything worked as intended, such as loading the model correctly and returning predictions. After that, we packaged the API together with the trained model into a Docker container, which we then tried to deploy to the Google Cloud platform.

We also made a frontend using Streamlit, which allows users to use our API and get a prediction based on an Image that the user uploads. This allows a very user-friendly way of using our model and API without any technical know-how.

### Question 25

> **Did you perform any unit testing and load testing of your API? If yes, explain how you did it and what results for**
> **the load testing did you get. If not, explain how you would do it.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *For unit testing we used ... and for load testing we used ... . The results of the load testing showed that ...*
> *before the service crashed.*
>
> Answer:

We did performed basic unit testing of our API, but we did not do a formal load testing. Unit tests were performed using pytest and focused on making sure that the API could start correctly and respond to requests in an expected way. In particular, we tested the inference endpoint by doing sample requests and veryfying that the response format and status code were correct. This helped us to make sure that model loading, request handling, and inference worked together as intended.

We did not perform systematic load testing, as got to the point of proof of concept, and did not test how it performed under high traffic with load tests.

### Question 26

> **Did you manage to implement monitoring of your deployed model? If yes, explain how it works. If not, explain how**
> **monitoring would help the longevity of your application.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We did not manage to implement monitoring. We would like to have monitoring implemented such that over time we could*
> *measure ... and ... that would inform us about this ... behaviour of our application.*
>
> Answer:

We did not set up dedicated monitoring for the deployed API with things like inference performance, response time, or data drift. If the project were extended further, monitoring would be important to make sure the application keeps working well over time. For example, monitoring could be used to keep an eye on request latency, error rates, and the types of predictions the model is making, and to notice if something starts to behave differently than expected. Combined with simple logging and alerts, this would make it easier to spot problems early, understand what is going on, and decide when the model needs to be retrained or the service needs to be scaled. This kind of monitoring would help make the deployed application more stable and reliable in the long run.

## Overall discussion of project

> In the following section we would like you to think about the general structure of your project.

### Question 27

> **How many credits did you end up using during the project and what service was most expensive? In general what do**
> **you think about working in the cloud?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *Group member 1 used ..., Group member 2 used ..., in total ... credits was spend during development. The service*
> *costing the most was ... due to ... . Working in the cloud was ...*
>
> Answer:

We actually worked on two Google Cloud Platforms simultaneously which was propably not ideal, but two team members tried it out on their own first, and then we included more team members afterwards, but had some difficulties in setting up a method for fast access. We used app. 21 credits, and the compute engine accounted for about 90% of the expenses. There seems to be a lot of potential in the cloud, especially when collaborating in groups, and for running heavy code. However, it can take some time to overcome the initial phase of setting the pipeline up. For future DL/ML projects it is definitely a platform that several team members consider using again. Moreover, it is an easy ad cheap way to uptaining computing power and storage space etc. all in one place. The only, dowside is being dependant on big tech companies, which will then hold power over your data.

### Question 28

> **Did you implement anything extra in your project that is not covered by other questions? Maybe you implemented**
> **a frontend for your API, use extra version control features, a drift detection service, a kubernetes cluster etc.**
> **If yes, explain what you did and why.**
>
> Recommended answer length: 0-200 words.
>
> Example:
> *We implemented a frontend for our API. We did this because we wanted to show the user ... . The frontend was*
> *implemented using ...*
>
> Answer:

We did implement a frotend for our API. A very simple frontend with the opportunity of uploading an image of your prefered beverages withing the three classes beer/wine/whiskey, and it will output the the predicted class based on our model in a string-format, and the probability of each class.


### Question 29

> **Include a figure that describes the overall architecture of your system and what services that you make use of.**
> **You can take inspiration from [this figure](figures/overview.png). Additionally, in your own words, explain the**
> **overall steps in figure.**
>
> Recommended answer length: 200-400 words
>
> Example:
>
> *The starting point of the diagram is our local setup, where we integrated ... and ... and ... into our code.*
> *Whenever we commit code and push to GitHub, it auto triggers ... and ... . From there the diagram shows ...*
>
> Answer:
ETETSTE
--- question 29 fill here ---

### Question 30

> **Discuss the overall struggles of the project. Where did you spend most time and what did you do to overcome these**
> **challenges?**
>
> Recommended answer length: 200-400 words.
>
> Example:
> *The biggest challenges in the project was using ... tool to do ... . The reason for this was ...*
>
> Answer:

Github: There were different levels of expereience with Git in the group in general, and only few with collaborative experience in Git, so there were some struggles on merging branches and overwritting/deleting files between group members.

Google Cloud Platform: None of us had expereience in using GCP, so there has a steep learning curve in getting aquainted with GCP, and also developing a system for easy use on one project among all members of the group.

Utilizing multiple frameworks siultaneously: In the project we tried out multiple frameworks for version control, debbugging, optimizing etc. and different members had the "lead" on different frameworks. This meant that configuring files, dependencies, requirements etc. was something that had to be communicated among members, and could cause bugs while running the code. Moreover, there was a challenge in getting used to a lot of new frameworks simultaneously and learning how to utilize and incorporating them in the code, and doing that as a group.

Config files and requirements: There were a few struggles related to incorporating new code, but not necessarily updating requirement files or config files.

Remote work: Much of the work was done remotely, which was introducing a further challenges, especially given the time-scope of the project. However, it did force us to ensure good coding practice, describe our code to be easily understood, and be very focussed on version control.




### Question 31

> **State the individual contributions of each team member. This is required information from DTU, because we need to**
> **make sure all members contributed actively to the project. Additionally, state if/how you have used generative AI**
> **tools in your project.**
>
> Recommended answer length: 50-300 words.
>
> Example:
> *Student sXXXXXX was in charge of developing of setting up the initial cookie cutter project and developing of the*
> *docker containers for training our applications.*
> *Student sXXXXXX was in charge of training our models in the cloud and deploying them afterwards.*
> *All members contributed to code by...*
> *We have used ChatGPT to help debug our code. Additionally, we used GitHub Copilot to help write some of our code.*
> Answer:

Student ralor: contributed to developing core modules train.py, data.py etc. He has contributed to developing FastApi, frontend integration, and GCP integration of the proejct. He has also contributed to report writing.
Student s204151: helped ensure high code quality by writing unit tests, using profiling to optimize code, adding logging events and calculating code coverage. Also participated in report writing.
Student s242875: Supported the development process by reviewing code structure, helping align the project with course modules, and assisting with documentation and report refinement. Also helped by testing components locally and verifying configurations.



AI has been used as for several parts of the project. It has been used as a step after prototyping, to debug code pieces of code, and to optimize the it as well. It has also been used partly to write easily understandable inline comments, and to standardize them. It has also been used to reformulate parts of the the report writing in order to make it sound more professional, and to reduce redundacy and increase readability. 
