## Stable Diffusion-driven Procedural Content Generation in an image-based English Vocabulary Learning Application
The application utilises [Stable Diffusion](https://github.com/AUTOMATIC1111/stable-diffusion-webui) for dynamically generating images from textual prompts and [WD14 Tagger](https://github.com/picobyte/stable-diffusion-webui-wd14-tagger) to produce descriptive tags relevant to the generated images.

Features:
* Content diversity through support for multiple SD checkpoint models.
* Filtering mechanisms to ensure appropriate content.
* Multiple difficulty levels and image topics/categories.
* Player rankings & leaderboard.
* Word definition lookup.
* Player statistics & question history.
* User feedback ticketing system.
* Remotely accessible via web browsers on any device.

<img src="https://github.com/kashyl/SD-based-English-Vocabulary-Acquisition/assets/44536334/c785c7eb-df81-4607-aa7c-3dbc84468c63" width="900">
<img src="https://github.com/kashyl/SD-based-English-Vocabulary-Acquisition/assets/44536334/a708a031-9f3b-4fba-816d-4bddf34b46cc">
<img src="https://github.com/kashyl/SD-based-English-Vocabulary-Acquisition/assets/44536334/0327ac60-dd3f-4e56-9731-a241fae1d0bf">

| Attribute                   | Value                        |
|-----------------------------|------------------------------|
| **Title**                   | SD-based-English-Vocabulary-Acquisition |
| **App File**                | main.py                      |
| **SDK**                     | gradio                       |
| **SDK Version**             | 3.39.0                       |

## How to run
1. Ensure your machine has a powerful enough GPU to run Stable Diffusion 1.4; [follow the instructions](https://github.com/AUTOMATIC1111/stable-diffusion-webui#installation-and-running) and set up SD on your machine.
2. Check `shared.py` for the models used in the project app; download the ones you’d like to use, such as *Realistic Vision v4.0* (`realisticVisionV40_v40VAE.safetensors [e9d3cedc4b]`), from a platform such as **HuggingFace** or **CivitAI**. Make sure the versions and hashes match (or change the checkpoint names in `shared.py` to suit yours). Add the models to SD as per instructions found on the repo page.
3. Download the `easynegative`, `bad_prompt_version2`, `ng_deepnegative_v1_75t` and `badhandsv5-neg` negative prompt embeddings (textual inversion) and add them to SD.
4. Run Stable Diffusion (should be on the default port, `127.0.0.1:7860`). With Stable Diffusion running in the background, run the app through `main.py` (after installing all the required packages – **might take a while**).
5. For user feedback functionality, create a Gmail account with 2FA enabled, and create an app pass (16 chars), then add `GMAIL_ADDRESS` and `GMAIL_APP_PASS` to the `.env` file.
6. For tag wiki search, create an account on a Booru image board such as `safebooru.donmai.us`, and create an API Key with permissions to the tag wiki, then add `DANBOORU_API_KEY` and `DANBOORU_USER_NAME`. Important: confirm the chosen image board has filtered content before proceeding (the .env “danbooru” names are legacy).
7. Get the WD14 Tagger required modules from the [bmaltais Kohya SS GUI repo](https://github.com/bmaltais/kohya_ss/tree/master): `huggingface_util.py`, `lpw_stable_diffusion.py`, `model_util.py`, `train_util.py`, `utils.py` and place them in `./library`.
8. When running the app in share mode (check debug/share settings in `main.py`), a link will be created, to which you can connect from any device. Before accessing the app, the auth screen will ask for a user and pass; set this up as `APP_AUTH_USER` and `APP_AUTH_PWD` in the `.env` file.
9. Run Stable Diffusion in the background, then start the app through `main.py`.
10. Access the app on http://127.0.0.1:7868/ or the generated share link (if `share=True` in `main.py`) on any device with a web browser.
11. Check the instructions on how to play the game/use the app (note: when `share=True`, app auth user & password will differ depending on what you’ve set them to).

## How to play (legacy, participant instructions)
#### 1. Accessing the App
* Open the provided link, which will take you to a login screen.
* Use the following credentials:
  * Username: "demo"
  * Password:
* Once logged in, you will find yourself on the “Main” tab.
#### 2. Setting up Your Account:
* Navigate to the "Account" section.
* Register a new account.
* After registration, you will be automatically logged into your new account.
#### 3. Playing the Game:
* On the “Main” tab, set the prompt category (the default is a random category) and your desired difficulty.
* Click "generate new image." An image and its associated tags will appear.
  * Note: If an inappropriate image is generated, you have the option to generate a new one or reveal and continue.
  * Please note that the game might occasionally make errors in its tag assessments.
* Select any tags or words related to the image. Aim to identify as many relevant tags as possible while avoiding irrelevant ones.
* Once done, click "submit." You will then be presented with the results. You can further explore details about a specific tag or word if you wish.
#### 4. Customizing Image Generation:
* You can modify the type of image generation model under the "Settings" tab. Options include Realistic, Cartoon, Drawing, Anime, Artistic
#### 5. Leaderboard:
* Under "Leaderboards", you can view player rankings.
#### 6. Account Details:
* Within the "Account" tab, you can:
  * Monitor your player statistics.
  * Revisit prior questions, which also highlight the tag relevance.
  * Browse a gallery of previously generated images.
  * If certain images are obscured because of their content rating, the visibility setting can be adjusted in the "Settings" tab.
#### 7. Completion:
* Engage in at least 10 questions. Feel free to experiment with various prompt categories and models.
* Once done, please fill out the following questionnaire: Experiment Feedback Form
