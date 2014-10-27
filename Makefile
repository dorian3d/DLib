
all:  lib

lib: DUtils/libDUtils.so DUtilsCV/libDUtilsCV.so/ DVision/libDVision.so
	mkdir -p lib/ && cp $^ lib/

DUtils/libDUtils.so:
	make -C DUtils

DUtilsCV/libDUtilsCV.so:
	make -C DUtilsCV

DVision/libDVision.so:
	make -C DVision

install: all
	make -C DUtils install && make -C DUtilsCV install && make -C DVision install

clean:
	make -C DUtils clean && make -C DUtilsCV clean && make -C DVision clean && \
	rm ./lib/*.so

uninstall:
	make -C DUtils uninstall && \
	make -C DUtilsCV uninstall && \
	make -C DVision uninstall

