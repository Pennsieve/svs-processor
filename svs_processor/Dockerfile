# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# TEST IMAGE
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

FROM pennsieve/base-processor-pandas-test:1-5800e81 as test

RUN apk update && apk upgrade

RUN apk add build-base \
        autoconf \
        automake \
        libtool \
        zlib-dev \
        libxml2-dev \
        jpeg-dev \
        openjpeg-dev \
        tiff-dev \
        glib-dev \
        gdk-pixbuf-dev \
        sqlite-dev \
        libjpeg-turbo-dev \
        libexif-dev \
        lcms2-dev \
        fftw-dev \
        giflib-dev \
        libpng-dev \
        libwebp-dev \
        orc-dev \
        poppler-dev \
        librsvg-dev \
        libgsf-dev \
        openexr-dev \
        gtk-doc \
        openssl-dev

RUN apk add --no-cache --update \
  --repository=http://dl-cdn.alpinelinux.org/alpine/v3.12/community \
  'hdf5' \
  'hdf5-dev'

COPY openslide-init.patch /tmp

RUN wget -O- https://github.com/openslide/openslide/releases/download/v3.4.1/openslide-3.4.1.tar.gz | tar xzC /tmp
RUN cd /tmp \
        && patch -p0 </tmp/openslide-init.patch \
        && cd openslide-3.4.1 \
        && ./configure --prefix=/usr --disable-static \
        && make \
        && make install

RUN pip install --upgrade pip
RUN pip install --no-cache-dir attrs==18.2.0 \
                               pyaml==18.11.0 \
                               moto==1.3.7 \
                               pillow==5.3.0 \
                               cryptography==2.4.2 \
                               pylint==1.9.3 \
                               pytest==4.0.1 \
                               pytest-cov==2.6.0 \
                               openslide-python==1.1.1 \
                               botocore==1.12.61 \
                               configparser==3.5.0 \
                               coverage==4.5.2 \
                               docker==3.6.0 \
                               docker-pycreds==0.4.0 \
                               mock==2.0 \
                               werkzeug==0.14.1


# code: processor
COPY base_image_microscopy_processor        /app/base_image_microscopy_processor
COPY svs_processor/svs_processor            /app/svs_processor
COPY svs_processor/run.py                   /app/run.py

# code: tests
COPY svs_processor/tests                    /app/tests

WORKDIR /app

ENTRYPOINT [""]


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# PRODUCTION IMAGE
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

FROM pennsieve/base-processor-pandas:1-5800e81 as prod

RUN apk update && apk upgrade

RUN apk add build-base \
        autoconf \
        automake \
        libtool \
        zlib-dev \
        libxml2-dev \
        jpeg-dev \
        openjpeg-dev \
        tiff-dev \
        glib-dev \
        gdk-pixbuf-dev \
        sqlite-dev \
        libjpeg-turbo-dev \
        libexif-dev \
        lcms2-dev \
        fftw-dev \
        giflib-dev \
        libpng-dev \
        libwebp-dev \
        orc-dev \
        poppler-dev \
        librsvg-dev \
        libgsf-dev \
        openexr-dev \
        gtk-doc \
        openssl-dev

COPY --from=test /usr/bin /usr/bin
COPY --from=test /usr/lib /usr/lib

RUN pip install --upgrade pip
RUN pip install --no-cache-dir attrs==18.2.0 \
                               pyaml==18.11.0 \
                               moto==1.3.7 \
                               pillow==5.3.0 \
                               cryptography==2.4.2 \
                               pylint==1.9.3 \
                               pytest==4.0.1 \
                               pytest-cov==2.6.0 \
                               openslide-python==1.1.1 \
                               botocore==1.12.61 \
                               configparser==3.5.0 \
                               coverage==4.5.2 \
                               docker==3.6.0 \
                               docker-pycreds==0.4.0 \
                               mock==2.0 \
                               werkzeug==0.14.1

# code: processor
COPY base_image_microscopy_processor        /app/base_image_microscopy_processor
COPY svs_processor/svs_processor            /app/svs_processor
COPY svs_processor/run.py                   /app/run.py

# code: tests
COPY svs_processor/tests                    /app/tests

WORKDIR /app

ENTRYPOINT [""]
