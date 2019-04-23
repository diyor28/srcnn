def create_model():
    input_tensor = Input(shape=(None, None, 3))

    conv = Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
    conv_0 = Conv2D(96, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(conv)
    conv_1 = Conv2D(128, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(conv_0)
    conv_1 = BatchNormalization()(conv_1)

    conv_2 = Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu')(conv_1)
    conv_2 = Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu')(conv_2)
    conv_2 = Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu')(conv_2)
    conv_2 = BatchNormalization()(conv_2)

    conc_0 = Concatenate()([conv_2, conv_1])

    conv_3 = UpSampling2D()(conc_0)
    conv_3 = Conv2D(96, kernel_size=(3, 3), padding='same', activation='relu')(conv_3)
    conv_3 = Conv2D(96, kernel_size=(3, 3), padding='same', activation='relu')(conv_3)
    conv_3 = BatchNormalization()(conv_3)

    conc_1 = Concatenate()([conv_3, conv_0])

    conv_4 = UpSampling2D()(conc_1)
    conv_4 = Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu')(conv_4)
    conv_4 = Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu')(conv_4)
    conv_4 = BatchNormalization()(conv_4)

    conc_2 = Concatenate()([conv_4, conv])

    conv_5 = Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu')(conc_2)
    conv_5 = Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu')(conv_5)
    conv_5 = BatchNormalization()(conv_5)

    conc_2 = Concatenate()([conv_5, conc_2])

    output = Conv2D(3, (3, 3), padding='same', activation=None)(conc_2)

    model = Model(inputs=[input_tensor], outputs=[output])
    model.compile(optimizer=Adam(lr=args.lr), loss='mse', metrics=[PSNRLoss])
    model.summary()
    return model
