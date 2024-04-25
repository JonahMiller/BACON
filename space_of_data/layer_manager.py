from space_of_data.layer_methods.bacon3 import BACON_3_layer
from space_of_data.layer_methods.gp_ranking import ranking_layer
from space_of_data.layer_methods.popularity import popular_layer


def layer_main(layer_type, kwargs):
    if kwargs:
        if layer_type == "bacon.3":
            return lambda df, layer_method, afs: BACON_3_layer(df, layer_method, afs, **kwargs)
        elif layer_type == "gp_ranking":
            return lambda df, layer_method, afs: ranking_layer(df, layer_method, afs, **kwargs)
        elif layer_type == "popularity":
            return lambda df, layer_method, afs: popular_layer(df, layer_method, afs, **kwargs)

    else:
        if layer_type == "bacon.3":
            return BACON_3_layer
        elif layer_type == "gp_ranking":
            return ranking_layer
        elif layer_type == "popularity":
            return popular_layer
