import json
import os
import re
from unicodedata import normalize

import PIL
from PIL import Image
import boto3
import numpy as np
from base_processor.imaging import utils
from botocore.client import Config
from base_processor.imaging.deepzoom import _get_or_create_path, _get_files_path
from openslide import ImageSlide, open_slide, OpenSlide, OpenSlideError, \
    deepzoom, PROPERTY_NAME_MPP_X, PROPERTY_NAME_MPP_Y
from base_image_microscopy_processor import BaseMicroscopyImageProcessor


def slugify(text):
    text = normalize('NFKD', text.lower()).encode('ascii', 'ignore').decode()
    return re.sub('[^a-z0-9]+', '-', text)

class SVSProcessorOutputs(object):
    def __init__(self, *args, **kwargs):
        self.slide = None
        self.associated_images = []
        self.associated_image_names = []
        self.ImageCount = kwargs.get('ImageCount', 1)

        self.img_dimensions = {}
        self.num_dimensions = -1
        self.file_path = None
        self.view_path = None
        self.ometiff_path = None

        self.img_data = kwargs.get('img_data', None)
        self.img_data_dtype = kwargs.get('img_data_dtype', None)
        self.metadata = kwargs.get('metadata', {})

        self.img_rdr = kwargs.get('img_rdr', None)
        self.DimensionOrder = kwargs.get('DimensionOrder', None)
        self.ImageCount = kwargs.get('ImageCount', -1)
        self.SizeX = kwargs.get('SizeX', -1)
        self.SizeY = kwargs.get('SizeY', -1)
        self.SizeZ = kwargs.get('SizeZ', -1)
        self.SizeC = kwargs.get('SizeC', -1)
        self.SizeT = kwargs.get('SizeT', -1)
        self.PixelType = kwargs.get('PixelType', None)
        self.RGBChannelCount = kwargs.get('RGBChannelCount', -1)

        self.isRGB = kwargs.get('isRGB', False)
        self.RGBDimension = kwargs.get('RGBDimension', -1)
        self.hasTimeDimension = kwargs.get('hasTimeDimension', False)
        self.TimeDimension = kwargs.get('TimeDimension', -1)

        self.view_format = 'dzi'
        self.optimize = kwargs.get('optimize', False)
        self.tile_size = kwargs.get('tile_size', 128)
        self.tile_overlap = kwargs.get('tile_overlap', 0)
        self.tile_format = kwargs.get('tile_format', "png")
        self.image_quality = kwargs.get('image_quality', 1.0)
        self.resize_filter = kwargs.get('resize_filter', "bicubic")

    def _load_and_save_assets(self, i_xyzct, n_xyzct):
        """Load image using OpenSlide"""

        # Parse parallelization arguments
        i_x, i_y, i_z, i_c, i_t = i_xyzct
        n_x, n_y, n_z, n_c, n_t = n_xyzct

        if i_x == -1 or n_x == -1:
            i_x = 0
            n_x = 1
        if i_y == -1 or n_y == -1:
            i_y = 0
            n_y = 1
        if i_z == -1 or n_z == -1:
            i_z = 0
            n_z = 1
        if i_c == -1 or n_c == -1:
            i_c = 0
            n_c = 1
        if i_t == -1 or n_t == -1:
            i_t = 0
            n_t = 1

        # Make view directory
        if not os.path.exists('%s-zoomed' % os.path.basename(self.file_path)):
            os.makedirs('%s-zoomed' % os.path.basename(self.file_path))
        asset_format = self.view_format

        try:
            self.slide = open_slide(self.file_path)
            z = 0
            t = 0
            for name, image in self.slide.associated_images.items():
                if 'thumbnail' in name:
                    continue
                # Increment z index
                z += 1
                self.associated_image_names.append(slugify(name))
                self.associated_images.append(ImageSlide(image))

                # Save thumbnails of associated images for slide viewer stack viewer
                if (i_x, i_y) == (0, 0) or (i_x, i_y) == (-1, -1):
                    timage = image.copy()
                    timage.thumbnail((200, 200), PIL.Image.ANTIALIAS)
                    timage.save(
                        os.path.join(
                            '%s-zoomed' % os.path.basename(self.file_path),
                            'dim_Z_slice_{slice_z}_dim_T_slice_{slice_t}_thumbnail.png'.format(
                                slice_z=z,
                                slice_t=t
                            )
                        )
                    )
                    # Create large thumbnail
                    timage = image.copy()
                    timage.thumbnail((1000, 1000), PIL.Image.ANTIALIAS)
                    timage.save(
                        os.path.join(
                            '%s-zoomed' % os.path.basename(self.file_path),
                            'dim_Z_slice_{slice_z}_dim_T_slice_{slice_t}_large_thumbnail.png'.format(
                                slice_z=z,
                                slice_t=t
                            )
                        )
                    )

                    filename = os.path.join(
                        '%s-zoomed' % os.path.basename(self.file_path),
                        'dim_Z_slice_{slice_z}_dim_T_slice_{slice_t}.{fmt}'.format(
                            slice_z=z, slice_t=t, fmt=asset_format)
                    )

                    utils.save_asset(
                        image.convert("RGB"),
                        asset_format,
                        filename,
                        optimize=self.optimize, tile_size=self.tile_size,
                        tile_overlap=self.tile_overlap, tile_format=self.tile_format,
                        image_quality=self.image_quality, resize_filter=self.resize_filter
                    )
        except OpenSlideError:
            raise OpenSlideError("file is recognized by OpenSlide but an error occurred")
        except IOError:
            raise IOError("File is not recognized at all")

        # Generate tile assets
        try:
            if isinstance(self.slide, ImageSlide) or isinstance(self.slide, OpenSlide):
                z = 0
                t = 0
                filename = os.path.join(
                    '%s-zoomed' % os.path.basename(self.file_path),
                    'dim_Z_slice_{slice_z}_dim_T_slice_{slice_t}.{fmt}'.format(
                        slice_z=z, slice_t=t, fmt=asset_format)
                )

                # Check to see if processor should process entire image view instead of just a sub-region
                if n_y == -1 or n_x == -1 \
                        or i_x == -1 or i_y == -1:
                    # Read ENTIRE image into memory
                    self.LOGGER.warn('Reading entire image {} into memory for encoding into DeepZoom'.format(self.file_path))
                    image = self.slide.read_region((0, 0), 0, self.slide.dimensions).convert("RGB")

                    # Encode into DZI
                    utils.encode_dzi(image, filename, tile_size=self.tile_size, tile_overlap=self.tile_overlap,
                                     tile_format=self.tile_format, image_quality=self.image_quality)
                else:
                    # Initialize DeepZoomGenerator from OpenSlide Python API
                    deepzoomgenerator = deepzoom.DeepZoomGenerator(osr=self.slide, tile_size=self.tile_size,
                                                                   overlap=self.tile_overlap)
                    # Create tiles using OpenSlide
                    image_files = _get_or_create_path(_get_files_path(filename))
                    for level in xrange(deepzoomgenerator.level_count):
                        level_dir = _get_or_create_path(os.path.join(image_files, str(level)))

                        # Determine number of rows and columns of tiles at this level
                        num_columns = deepzoomgenerator.level_tiles[level][0]
                        num_rows = deepzoomgenerator.level_tiles[level][1]

                        # Iterate over all tiles to determine which ones fall in svs-processor sub-region
                        for column in xrange(num_columns):
                            for row in xrange(num_rows):
                                # Calculate the appropriate sub-region column and row index for this tile
                                try:
                                    tile_i_x = int(column / (float(num_columns) / n_x))
                                except ZeroDivisionError:
                                    tile_i_x = 0
                                try:
                                    tile_i_y = int(row / (float(num_rows) / n_y))
                                except ZeroDivisionError:
                                    tile_i_y = 0

                                # Check if this tile falls in the given sub-region
                                if tile_i_x == i_x and \
                                        tile_i_y == i_y:
                                    # Get tile image and save it
                                    tile = deepzoomgenerator.get_tile(level, (column, row))
                                    format = self.tile_format
                                    tile_path = os.path.join(level_dir,
                                                             '%s_%s.%s' % (column, row, format))
                                    tile_file = open(tile_path, 'wb')
                                    if self.tile_format == 'jpg':
                                        jpeg_quality = int(self.image_quality * 100)
                                        tile.save(tile_file, 'JPEG', quality=jpeg_quality)
                                    else:
                                        tile.save(tile_file)
                    # Create descriptor only once; defaulted to when we're at the very first sub-region
                    if i_y == 0 and i_x == 0:
                        open(filename, 'w').write(deepzoomgenerator.get_dzi(self.tile_format))

                        # Generate and save thumbnail
                        timage = self.slide.get_thumbnail((200, 200))
                        timage.save(
                            os.path.join(
                                '%s-zoomed' % os.path.basename(self.file_path),
                                'dim_Z_slice_0_dim_T_slice_0_thumbnail.png'
                            )
                        )

                        # Generate properties metadata.json metadata
                        metadata = []
                        for property_key, property_value in self.slide.properties.items():
                            property = {}
                            property["key"] = property_key
                            property["value"] = property_value
                            property["dataType"] = "String"
                            property["category"] = "Blackfynn"
                            property["fixed"] = False
                            property["hidden"] = False
                            metadata.append(property)
                        with open('metadata.json','w') as fp:
                            json.dump(metadata, fp)
                self.view_path = os.path.join(os.getcwd(), '%s-zoomed' % os.path.basename(self.file_path))
        except OpenSlideError:
            pass
        return

    @property
    def file_size(self):
        """Return file size"""
        return os.path.getsize(self.file_path)

    @property
    def view_size(self):
        return os.path.getsize(self.view_path)

    def get_view_asset_dict(self, storage_bucket, upload_key):
        upload_key = upload_key.rstrip('/')
        json_dict = {
            "bucket": storage_bucket,
            "key": upload_key,
            "type": "View",
            "size": self.view_size,
            "fileType": "Image"
        }
        return json_dict

    def get_dim_assignment(self):
        """Retrieve inferred dimension assignment based on number of dimensions and length of each dimension. """
        return self.DimensionOrder

    def set_img_properties(self, asset_format='dzi'):
        """Create and assign properties for the image"""

        self.view_format = asset_format

        # Load image using openslide
        try:
            self.slide = open_slide(self.file_path)
            assert isinstance(self.slide, ImageSlide) or isinstance(self.slide, OpenSlide)
            z_index = 0
            for name, image in self.slide.associated_images.items():
                if 'thumbnail' in name:
                    continue
                # Increment z index
                z_index += 1
                self.associated_image_names.append(slugify(name))
                self.associated_images.append(ImageSlide(image))
        except OpenSlideError:
            raise OpenSlideError("file is recognized by OpenSlide but an error occurred")
        except IOError:
            raise IOError("File is not recognized at all")

        # Get image data details
        self.DimensionOrder = list('XYZCT')
        self.ImageCount = len(self.slide.associated_images.values()) + 1
        self.SizeX = self.slide.dimensions[0]
        self.SizeY = self.slide.dimensions[1]
        self.SizeC = 3
        self.SizeT = 1
        self.SizeZ = len(self.associated_images) + 1
        self.PixelType = 1
        self.RGBChannelCount = 3
        self.isRGB = True

        self.RGBDimension = 3
        self.hasTimeDimension = True
        self.TimeDimension = 4

        # Set data type
        if self.PixelType == 0:  # int8
            # self.img_data = self.img_data.astype(np.int8)
            self.img_data_dtype = np.int8
        elif self.PixelType == 1:  # uint8
            # self.img_data = self.img_data.astype(np.uint8)
            self.img_data_dtype = np.uint8
        elif self.PixelType == 2:  # int16
            # self.img_data = self.img_data.astype(np.int16)
            self.img_data_dtype = np.int16
        elif self.PixelType == 3:  # uint16
            # self.img_data = self.img_data.astype(np.uint16)
            self.img_data_dtype = np.uint16
        elif self.PixelType == 4:  # int32
            # self.img_data = self.img_data.astype(np.int32)
            self.img_data_dtype = np.int32
        elif self.PixelType == 5:  # uint32
            # self.img_data = self.img_data.astype(np.uint32)
            self.img_data_dtype = np.uint32
        elif self.PixelType == 6:  # float
            # self.img_data = self.img_data.astype(np.float)
            self.img_data_dtype = np.float
        elif self.PixelType == 7:  # double
            # self.img_data = self.img_data.astype(np.double)
            self.img_data_dtype = np.double

        # Set number of dimensions of image matrix
        self.num_dimensions = 5
        self.img_data_shape = [self.SizeX, self.SizeY, self.SizeZ, self.SizeC, self.SizeT]

        dim_assignment = list('XYZCT')  # Force assignment

        self.img_dimensions['filename'] = os.path.basename(self.file_path)
        self.img_dimensions['num_dimensions'] = self.num_dimensions
        self.img_dimensions['isColorImage'] = False
        self.img_dimensions['dimensions'] = {}

        for dim in range(self.num_dimensions):
            self.img_dimensions['dimensions'][dim] = {}
            self.img_dimensions['dimensions'][dim]["assignment"] = dim_assignment[dim]
            self.img_dimensions['dimensions'][dim]["length"] = self.img_data_shape[dim]
            self.img_dimensions['dimensions'][dim]["resolution"] = -1
            self.img_dimensions['dimensions'][dim]["units"] = "um"
            if dim_assignment[dim] == 'C' and self.isRGB:
                self.RGBDimension = dim
            if dim_assignment[dim] == 'T':
                self.hasTimeDimension = True
                self.TimeDimension = dim
        self.img_dimensions['isColorImage'] = self.isRGB

        # Get metadata
        metadata = self.slide.properties

        # Get resolution
        try:
            self.img_dimensions['dimensions'][0]["resolution"] = metadata[PROPERTY_NAME_MPP_X]
        except ValueError:
            self.img_dimensions['dimensions'][0]["resolution"] = -1
        try:
            self.img_dimensions['dimensions'][1]["resolution"] = metadata[PROPERTY_NAME_MPP_Y]
        except ValueError:
            self.img_dimensions['dimensions'][1]["resolution"] = -1

        return


    def load_and_save_assets(self, svs_file_path, i_xyzct=(-1, -1, -1, -1, -1), n_xyzct=(-1, -1, -1, -1, -1),
                             asset_format='dzi'):
        """Load image and generate view assets"""

        # Set file path
        self.file_path = svs_file_path

        # Set image properties
        self.set_img_properties(asset_format=asset_format)

        # Load image
        self._load_and_save_assets(i_xyzct=i_xyzct, n_xyzct=n_xyzct)

    def save_ometiff(self):
        """Save OME-TIFF images derived from SVS file"""

        # Not implemented because of possible memory issues when loading entire 2D plane to write to OME TIFF
        raise NotImplementedError

class SVSProcessor(BaseMicroscopyImageProcessor):
    required_inputs = ['file']

    def __init__(self, *args, **kwargs):
        super(SVSProcessor, self).__init__(*args, **kwargs)
        self.session = boto3.session.Session()
        self.s3_client = self.session.client('s3', config=Config(signature_version='s3v4'))

        self.file_path = self.inputs.get('file')

        self.upload_key = None
        try:
            self.optimize = utils.str2bool(self.inputs.get('optimize_view'))
        except AttributeError:
            self.optimize = False

        try:
            self.tile_size = int(self.inputs.get('tile_size'))
        except (ValueError, KeyError, TypeError) as e:
            self.tile_size = 128

        try:
            self.tile_overlap = int(self.inputs.get('tile_overlap'))
        except (ValueError, KeyError, TypeError) as e:
            self.tile_overlap = 0

        try:
            self.tile_format = self.inputs.get('tile_format')
            if self.tile_format is None:
                self.tile_format = "png"
        except KeyError:
            self.tile_format = "png"

        try:
            self.image_quality = float(self.inputs.get('image_quality'))
        except (ValueError, KeyError, TypeError) as e:
            self.image_quality = 1.0

        try:
            self.resize_filter = self.inputs.get('resize_filter')
        except KeyError:
            self.resize_filter = "bicubic"

    def load_and_save(self, i_xyzct=(-1, -1, -1, -1, -1), n_xyzct=(-1, -1, -1, -1, -1)):
        """Run load_image and save OME-TIFF for all output files"""
        if os.path.isfile(self.file_path):
            output_file = SVSProcessorOutputs(optimize=self.optimize, tile_size=self.tile_size,
                                        tile_overlap=self.tile_overlap, tile_format=self.tile_format,
                                        image_quality=self.image_quality, resize_filter=self.resize_filter)
            output_file.load_and_save_assets(self.file_path, i_xyzct=i_xyzct, n_xyzct=n_xyzct)
            self.outputs.append(output_file)

    def task(self):
        """Run main task which will load and save view assets for all CZI output files deocded"""

        # self._load_image()
        self.LOGGER.info('Got inputs {}'.format(self.inputs))

        # Get sub_region index
        try:
            sub_region = self.inputs['sub_region_file']
            sub_region_regex = r'sub_' \
                               r'x_([0-9]+)_([0-9]+)_' \
                               r'y_([0-9]+)_([0-9]+)_' \
                               r'z_([0-9]+)_([0-9]+)_' \
                               r'c_([0-9]+)_([0-9]+)_' \
                               r't_([0-9]+)_([0-9]+).txt'
            i_x = int(re.match(re.compile(sub_region_regex), sub_region).groups()[0])
            n_x = int(re.match(re.compile(sub_region_regex), sub_region).groups()[1])
            i_y = int(re.match(re.compile(sub_region_regex), sub_region).groups()[2])
            n_y = int(re.match(re.compile(sub_region_regex), sub_region).groups()[3])
            i_z = int(re.match(re.compile(sub_region_regex), sub_region).groups()[4])
            n_z = int(re.match(re.compile(sub_region_regex), sub_region).groups()[5])
            i_c = int(re.match(re.compile(sub_region_regex), sub_region).groups()[6])
            n_c = int(re.match(re.compile(sub_region_regex), sub_region).groups()[7])
            i_t = int(re.match(re.compile(sub_region_regex), sub_region).groups()[8])
            n_t = int(re.match(re.compile(sub_region_regex), sub_region).groups()[9])
        except (KeyError, IndexError, AttributeError):
            i_x = -1
            n_x = -1
            i_y = -1
            n_y = -1
            i_z = -1
            n_z = -1
            i_c = -1
            n_c = -1
            i_t = -1
            n_t = -1

        # Load and save view images
        self.load_and_save(i_xyzct=(i_x, i_y, i_z, i_c, i_t), n_xyzct=(n_x, n_y, n_z, n_c, n_t))

        # Save dimensions object as JSON in view/ directory (for now)
        if os.path.isfile(self.file_path):
            # Output file is just the one and only file in outputs
            output_file = self.outputs[0]

            # Save dimensions object as JSON in view/ directory (for now)
            with open(os.path.join('%s-zoomed' % os.path.basename(self.file_path), 'dimensions.json'), 'w') as fp:
                json.dump(output_file.img_dimensions, fp)

            # Create create-asset JSON object file called view_asset_info.json
            self.upload_key = os.path.join(
                self.settings.storage_directory,
                os.path.basename(output_file.file_path) + '-zoomed'
            )

            with open('view_asset_info.json', 'w') as fp:
                json.dump(output_file.get_view_asset_dict(
                    self.settings.storage_bucket,
                    self.upload_key
                ), fp)

            # Generate properties metadata.json metadata
            metadata = []
            img_dimensions = self.outputs[0].img_dimensions
            for dim in range(self.outputs[0].num_dimensions):
                for property_key_suffix in ["assignment", "length", "resolution", "units"]:
                    # Initialize property
                    property = {}

                    # Set property key and value
                    property_key = 'dimensions.%i.%s' % (dim, property_key_suffix)
                    property_value = str(img_dimensions['dimensions'][dim][property_key_suffix])

                    # Create property instance
                    property["key"] = property_key
                    property["value"] = property_value
                    property["dataType"] = "String"
                    property["category"] = "Blackfynn"
                    property["fixed"] = False
                    property["hidden"] = False
                    metadata.append(property)
            for property_key, property_value in self.outputs[0].slide.properties.items():
                property = {}
                property["key"] = property_key
                property["value"] = property_value
                property["dataType"] = "String"
                property["category"] = "Blackfynn"
                property["fixed"] = False
                property["hidden"] = False
                metadata.append(property)
            with open('metadata.json', 'w') as fp:
                json.dump(metadata, fp)

            ## Hack for backwards compatibility for DeepZoom viewer
            os.system('cp -r %s-zoomed/dim_Z_slice_0_dim_T_slice_0.dzi %s-zoomed/slide.dzi' %
                      (os.path.basename(self.file_path), os.path.basename(self.file_path)))
            os.system('cp -r %s-zoomed/dim_Z_slice_0_dim_T_slice_0_files %s-zoomed/slide_files' %
                      (os.path.basename(self.file_path), os.path.basename(self.file_path)))
