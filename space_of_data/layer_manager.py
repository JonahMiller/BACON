from space_of_data.layer_methods.bacon3 import BACON_3_layer
from space_of_data.layer_methods.gp_ranking import ranking_layer
from space_of_data.layer_methods.popularity import popular_layer


def layer_main(layer_type, kwarg_dict):
    if kwarg_dict:
        if layer_type == "bacon.3":
            return lambda df, layer_method: BACON_3_layer(df, layer_method, **kwarg_dict)
        elif layer_type == "gp_ranking":
            return lambda df, layer_method: ranking_layer(df, layer_method, **kwarg_dict)
        elif layer_type == "popularity":
            return lambda df, layer_method: popular_layer(df, layer_method, **kwarg_dict)

    else:
        if layer_type == "bacon.3":
            return BACON_3_layer
        elif layer_type == "gp_ranking":
            return ranking_layer
        elif layer_type == "popularity":
            return popular_layer
