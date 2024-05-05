from space_of_data.layer_methods.layer_select import layer


def layer_main(layer_type, kwargs):
    if kwargs:
        return lambda df, laws_method, afs: layer(df, laws_method, afs, layer_type, **kwargs)

    else:
        return lambda df, laws_method, afs: layer(df, laws_method, afs, layer_type)
