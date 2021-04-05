# Kivy App

In an effort to minimize the overhead associated with porting, we decided to release our app in [kivy](https://kivy.org/#home).

Refer to the [wiki](https://github.com/yycho0108/PhoneBot/wiki/Deploying-Phonebot-App-on-Android) for instructions,

and also refer to the [Makefile](app/Makefile) for various pre-configured instructions accordingly.

## Key Content

- [Makefile](app/Makefile): Pre-configured build and deploy instructions.
- [buildozer.spec](app/buildozer.spec): Buildozer configuration.
- [buildozer.Dockerfile](app/buildozer.Dockerfile): Docker container configuration for building the app.
- [main.py](app/main.py): Main Kivy application entry-point.
- [receipes](app/recipes): Custom Python4Android recipes.

## Docker

Install docker compose:

```bash
sudo curl -L "https://github.com/docker/compose/releases/download/1.26.2/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

Run container:

```bash
cd app
docker-compose up
docker-compose exec kivy bash
```

To fix OpenCV build [issues](https://github.com/kivy/buildozer/issues/1144):

https://github.com/kivy/buildozer/issues/1144#issuecomment-655548082
