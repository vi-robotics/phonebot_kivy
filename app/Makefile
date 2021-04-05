all: build

.PHONY: docker docker-shell clean build install uninstall

docker:
	docker build --tag=buildozer ../../ -f buildozer.Dockerfile

docker-shell:
	docker run --privileged -v /dev/bus/usb:/dev/bus/usb --net host \
		--interactive --tty  --volume ${PWD}/.buildozer:/home/user/.buildozer \
		--volume ${PWD}/../..:/home/user/hostcwd \
		-w /home/user/hostcwd/App/KivyApp/ \
		--entrypoint /bin/bash buildozer \


clean:
	buildozer -v android clean

build:
	@if [ ! -d "${HOME}/.buildozer" ]; then mkdir -p "${HOME}/.buildozer"; fi;
	@sudo chown ${USER} -R "${HOME}/.buildozer"
	buildozer -v android debug

install:
	# NOTE(yycho0108): Adding service declaration to AndroidManifest template before deploying the app.
	@MANIFEST="./.buildozer/android/platform/build-armeabi-v7a/dists/phonebotapp__armeabi-v7a/templates/AndroidManifest.tmpl.xml"; \
		if ! grep -q 'DummyService' $${MANIFEST}; then sed -i '110i<service android:name="org.phonebot.DummyService"/>' $${MANIFEST}; fi;
	@MANIFEST="./.buildozer/android/platform/build-armeabi-v7a/dists/phonebotapp__armeabi-v7a/templates/AndroidManifest.tmpl.xml"; \
		if ! grep -q 'BluetoothLeService' $${MANIFEST}; then sed -i '111i<service android:name="main.com.phonebot.BluetoothLeService"/>' $${MANIFEST}; fi;
	buildozer -v android debug deploy run logcat

uninstall:
	buildozer -v android adb -- uninstall org.phonebot.phonebotapp
