import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, concatenate, BatchNormalization, PReLU, SpatialDropout2D, Add, \
    Conv2DTranspose, ReLU, Activation, Permute, ZeroPadding2D, UpSampling2D, Dense, Reshape, Concatenate

class ENET():

    def __init__(self, input_size, nclasses):
        """

        :param input_size: shape of the input image
        :param nclasses:  number of classes
        """
        self.im_width = input_size[0]
        self.im_height = input_size[1]
        self.nclasses = nclasses

    def initial_block(self, tensor):
        """
        Initial block operating on the input three channels of the image
        :param tensor: input tensor
        :return: the concatenated feature map
        """
        conv = Conv2D(filters=13, kernel_size=(3, 3), strides=(2, 2), padding='same', name='initial_block_conv',
                      kernel_initializer='he_normal')(tensor)
        pool = MaxPooling2D(pool_size=(2, 2), name='initial_block_pool')(tensor)
        #  concatenate the resulting feature maps
        concat = concatenate([conv, pool], axis=-1, name='initial_block_concat')
        return concat

    def bottleneck_encoder(self, tensor, nfilters, downsampling=False, dilated=False, asymmetric=False, normal=False,
                           drate=0.1,
                           name=''):

        """
        Encoder

        :param tensor: input tensor
        :param nfilters: Number of filters
        :param downsampling: Downsample the feature map
        :param dilated: determines  if ther should be dilated convultion
        :param asymmetric:  Determines if there should be asymmetric convolution
        :param normal:  enables 3x3 convolution on feature map
        :param drate: rate of dilation
        :param name: the name for the weight variable.
        :return: encoder output
        """
        y = tensor
        skip = tensor
        stride = 1
        ksize = 1

        # Filters operating on downsampled images have a bigger receptive field and hence gathers more context.
        if downsampling:
            stride = 2
            ksize = 2
            skip = MaxPooling2D(pool_size=(2, 2), name='max_pool_' + str(name))(skip)
            skip = Permute((1, 3, 2), name='permute_1_' + str(name))(skip)  # (B, H, W, C) -> (B, H, C, W)
            ch_pad = nfilters - tf.compat.v2.keras.backend.int_shape(tensor)[-1]
            skip = ZeroPadding2D(padding=((0, 0), (0, ch_pad)), name='zeropadding_' + str(name))(skip)
            skip = Permute((1, 3, 2), name='permute_2_' + str(name))(skip)  # (B, H, C, W) -> (B, H, W, C)

        y = Conv2D(filters=nfilters // 4, kernel_size=(ksize, ksize), kernel_initializer='he_normal',
                   strides=(stride, stride), padding='same', use_bias=False, name='1x1_conv_' + str(name))(y)
        y = BatchNormalization(momentum=0.1, name='bn_1x1_' + str(name))(y)
        y = PReLU(shared_axes=[1, 2], name='prelu_1x1_' + str(name))(y)

        if normal:
            # deconv with 3x3 filter
            y = Conv2D(filters=nfilters // 4, kernel_size=(3, 3), kernel_initializer='he_normal', padding='same',
                       name='3x3_conv_' + str(name))(y)
        elif asymmetric:
            # decompose 5x5 convolution to two asymmetric layers as 5x1 and 1x5
            y = Conv2D(filters=nfilters // 4, kernel_size=(5, 1), kernel_initializer='he_normal', padding='same',
                       use_bias=False, name='5x1_conv_' + str(name))(y)
            y = Conv2D(filters=nfilters // 4, kernel_size=(1, 5), kernel_initializer='he_normal', padding='same',
                       name='1x5_conv_' + str(name))(y)
        elif dilated:
            y = Conv2D(filters=nfilters // 4, kernel_size=(3, 3), kernel_initializer='he_normal',
                       dilation_rate=(dilated, dilated), padding='same', name='dilated_conv_' + str(name))(y)
        y = BatchNormalization(momentum=0.1, name='bn_main_' + str(name))(y)
        y = PReLU(shared_axes=[1, 2], name='prelu_' + str(name))(y)

        y = Conv2D(filters=nfilters, kernel_size=(1, 1), kernel_initializer='he_normal', use_bias=False,
                   name='final_1x1_' + str(name))(y)
        y = BatchNormalization(momentum=0.1, name='bn_final_' + str(name))(y)
        y = SpatialDropout2D(rate=drate, name='spatial_dropout_final_' + str(name))(y)

        y = Add(name='add_' + str(name))([y, skip])
        y = PReLU(shared_axes=[1, 2], name='prelu_out_' + str(name))(y)

        return y

    def bottleneck_decoder(self, tensor, nfilters, upsampling=False, normal=False, name=''):
        """

        :param tensor: input tensor
        :param nfilters: number of filters
        :param upsampling: Enables Transposed convolution
        :param normal: Enables 3x3 convolution on feature map
        :param name: The name for the weight variable.
        :return: decoder output
        """
        y = tensor
        skip = tensor
        if upsampling:
            skip = Conv2D(filters=nfilters, kernel_size=(1, 1), kernel_initializer='he_normal', strides=(1, 1),
                          padding='same', use_bias=False, name='1x1_conv_skip_' + str(name))(skip)
            skip = UpSampling2D(size=(2, 2), name='upsample_skip_' + str(name))(skip)

        y = Conv2D(filters=nfilters // 4, kernel_size=(1, 1), kernel_initializer='he_normal', strides=(1, 1),
                   padding='same', use_bias=False, name='1x1_conv_' + str(name))(y)
        y = BatchNormalization(momentum=0.1, name='bn_1x1_' + str(name))(y)
        y = PReLU(shared_axes=[1, 2], name='prelu_1x1_' + str(name))(y)

        if upsampling:
            # upsample with learned weights through convolution with a fractional stride
            y = Conv2DTranspose(filters=nfilters // 4, kernel_size=(3, 3), kernel_initializer='he_normal',
                                strides=(2, 2),
                                padding='same', name='3x3_deconv_' + str(name))(y)
        elif normal:
            Conv2D(filters=nfilters // 4, kernel_size=(3, 3), strides=(1, 1), kernel_initializer='he_normal',
                   padding='same', name='3x3_conv_' + str(name))(y)
        y = BatchNormalization(momentum=0.1, name='bn_main_' + str(name))(y)
        y = PReLU(shared_axes=[1, 2], name='prelu_' + str(name))(y)

        y = Conv2D(filters=nfilters, kernel_size=(1, 1), kernel_initializer='he_normal', use_bias=False,
                   name='final_1x1_' + str(name))(y)
        y = BatchNormalization(momentum=0.1, name='bn_final_' + str(name))(y)

        y = Add(name='add_' + str(name))([y, skip])
        y = ReLU(name='relu_out_' + str(name))(y)

        return y

    def build(self):
        """
            Build the model for training
        """

        print('. . . . .Building ENet. . . . .')

        img_input = Input(shape=(self.im_height, self.im_width, 3), name='image_input')

        x = self.initial_block(img_input)

        x = self.bottleneck_encoder(x, 64, downsampling=True, normal=True, name='1.0', drate=0.01)
        for i in range(1, 5):
            x = self.bottleneck_encoder(x, 64, normal=True, name='1.' + str(i), drate=0.01)

        # Encoder Block
        x = self.bottleneck_encoder(x, 128, downsampling=True, normal=True, name='2.0')
        x = self.bottleneck_encoder(x, 128, normal=True, name='2.1')
        x = self.bottleneck_encoder(x, 128, dilated=True, name='2.2')
        x = self.bottleneck_encoder(x, 128, asymmetric=True, name='2.3')
        x = self.bottleneck_encoder(x, 128, dilated=True, name='2.4')
        x = self.bottleneck_encoder(x, 128, normal=True, name='2.5')
        x = self.bottleneck_encoder(x, 128, dilated=True, name='2.6')
        x = self.bottleneck_encoder(x, 128, asymmetric=True, name='2.7')
        x = self.bottleneck_encoder(x, 128, dilated=True, name='2.8')

        x = self.bottleneck_encoder(x, 128, normal=True, name='3.0')
        x = self.bottleneck_encoder(x, 128, dilated=True, name='3.1')
        x = self.bottleneck_encoder(x, 128, asymmetric=True, name='3.2')
        x = self.bottleneck_encoder(x, 128, dilated=True, name='3.3')
        x = self.bottleneck_encoder(x, 128, normal=True, name='3.4')
        x = self.bottleneck_encoder(x, 128, dilated=True, name='3.5')
        x = self.bottleneck_encoder(x, 128, asymmetric=True, name='3.6')
        x = self.bottleneck_encoder(x, 128, dilated=True, name= '3.7')

        # Decoder Block
        x = self.bottleneck_decoder(x, 64, upsampling=True, name='4.0')
        x = self.bottleneck_decoder(x, 64, normal=True, name='4.1')
        x = self.bottleneck_decoder(x, 64, normal=True, name='4.2')

        x = self.bottleneck_decoder(x, 16, upsampling=True, name='5.0')
        x = self.bottleneck_decoder(x, 16, normal=True, name='5.1')

        img_output = Conv2DTranspose(self.nclasses, kernel_size=(2, 2), strides=(2, 2), kernel_initializer='he_normal',
                                     padding='same', name='image_output')(x)
        img_output = Activation('softmax')(img_output)

        model = Model(inputs=img_input, outputs=img_output, name='ENET')
        print('. . . . .Build Compeleted. . . . .')
        return model


class VGG():

    def __init__(self, input_size, nclasses):
        """

               :param input_size: shape of the input image
               :param nclasses:  number of classes
        """

        self.im_width = input_size[0]
        self.im_height = input_size[1]
        self.nclasses = nclasses

    def build(self):
        """
        Build the model for training
        """

        print('. . . . .Building VGG. . . . .')

        inputs = Input(shape=(self.im_height, self.im_width, 3))

        # Block 1
        block1_conv1 = Conv2D(
            64, (3, 3), activation='relu', padding='same',
            name='block1_conv1')(inputs)
        block1_conv2 = Conv2D(
            64, (3, 3), activation='relu', padding='same', name='block1_conv2')(block1_conv1)
        block1_pool = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(block1_conv2)

        # Block 2
        block2_conv1 = Conv2D(
            128, (3, 3), activation='relu', padding='same', name='block2_conv1')(block1_pool)
        block2_conv2 = Conv2D(
            128, (3, 3), activation='relu', padding='same', name='block2_conv2')(block2_conv1)
        block2_pool = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(block2_conv2)

        # Block 3
        block3_conv1 = Conv2D(
            256, (3, 3), activation='relu', padding='same', name='block3_conv1')(block2_pool)
        block3_conv2 = Conv2D(
            256, (3, 3), activation='relu', padding='same', name='block3_conv2')(block3_conv1)
        block3_conv3 = Conv2D(
            256, (3, 3), activation='relu', padding='same', name='block3_conv3')(block3_conv2)
        block3_pool = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(block3_conv3)

        # Block 4
        block4_conv1 = Conv2D(
            512, (3, 3), activation='relu', padding='same', name='block4_conv1')(block3_pool)
        block4_conv2 = Conv2D(
            512, (3, 3), activation='relu', padding='same', name='block4_conv2')(block4_conv1)
        block4_conv3 = Conv2D(
            512, (3, 3), activation='relu', padding='same', name='block4_conv3')(block4_conv2)
        block4_pool = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(block4_conv3)

        # Block 5
        block5_conv1 = Conv2D(
            512, (3, 3), activation='relu', padding='same', name='block5_conv1')(block4_pool)
        block5_conv2 = Conv2D(
            512, (3, 3), activation='relu', padding='same', name='block5_conv2')(block5_conv1)
        block5_conv3 = Conv2D(
            512, (3, 3), activation='relu', padding='same', name='block5_conv3')(block5_conv2)
        block5_pool = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(block5_conv3)

        pool5_conv1x1 = Conv2D(2, (1, 1), activation='relu', padding='same')(block5_pool)
        upsample_1 = Conv2DTranspose(2, kernel_size=(4, 4), strides=(2, 2), padding="same")(pool5_conv1x1)

        pool4_conv1x1 = Conv2D(2, (1, 1), activation='relu', padding='same')(block4_pool)
        add_1 = Add()([upsample_1, pool4_conv1x1])

        upsample_2 = Conv2DTranspose(2, kernel_size=(4, 4), strides=(2, 2), padding="same")(add_1)
        pool3_conv1x1 = Conv2D(2, (1, 1), activation='relu', padding='same')(block3_pool)
        add_2 = Add()([upsample_2, pool3_conv1x1])

        upsample_3 = Conv2DTranspose(2, kernel_size=(16, 16), strides=(8, 8), padding="same")(add_2)
        output = Dense(2, activation='softmax')(upsample_3)

        model = Model(inputs, output, name='multinet_seg')

        print('. . . . .Build Compeleted. . . . .')
        return model


class UNET():

    def __init__(self, input_size, nclasses):
        """

        :param input_size:  shape of the input image
        :param nclasses: number of classes
        """

        self.im_width = input_size[0]
        self.im_height = input_size[1]
        self.nclasses = nclasses

    def make_conv_block(self, nb_filters, input_tensor, block):
        """

        :param nb_filters: number of filters
        :param input_tensor: input tensor to perform convolution
        :param block: block number
        :return:
        """

        def make_stage(input_tensor, stage):
            name = 'conv_{}_{}'.format(block, stage)
            x = Conv2D(nb_filters, (3, 3), activation='relu',
                       padding='same', name=name)(input_tensor)
            name = 'batch_norm_{}_{}'.format(block, stage)
            x = BatchNormalization(name=name)(x)
            x = Activation('relu')(x)
            return x

        x = make_stage(input_tensor, 1)
        x = make_stage(x, 2)
        return x

    def build(self):
        """
        Build the model for training
        """
        print('. . . . .Building UNET. . . . .')

        inputs = Input(shape=(self.im_height, self.im_width, 3))
        conv1 = self.make_conv_block(32, inputs, 1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = self.make_conv_block(64, pool1, 2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = self.make_conv_block(128, pool2, 3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = self.make_conv_block(256, pool3, 4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        conv5 = self.make_conv_block(512, pool4, 5)

        up6 = Concatenate()([UpSampling2D(size=(2, 2))(conv5), conv4])
        conv6 = self.make_conv_block(256, up6, 6)

        up7 = Concatenate()([UpSampling2D(size=(2, 2))(conv6), conv3])
        conv7 = self.make_conv_block(128, up7, 7)

        up8 = Concatenate()([UpSampling2D(size=(2, 2))(conv7), conv2])
        conv8 = self.make_conv_block(64, up8, 8)

        up9 = Concatenate()([UpSampling2D(size=(2, 2))(conv8), conv1])
        conv9 = self.make_conv_block(32, up9, 9)

        conv10 = Conv2D(self.nclasses, (1, 1), name='conv_10_1')(conv9)

        x = Reshape((self.im_width * self.im_height, self.nclasses))(conv10)
        x = Activation('softmax')(x)
        outputs = Reshape((self.im_height, self.im_width, self.nclasses))(x)

        model = Model(inputs=inputs, outputs=outputs)

        print('. . . . .Build Compeleted. . . . .')

        return model
