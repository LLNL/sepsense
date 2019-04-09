import torch

class FMOW:
    def __init__(self, bg_flag=False, cat_list=None):
        if cat_list is None:
            category_list = ("airport,airport_hangar,airport_terminal,amusement_park,aquaculture,archaeological_site,\
                barn,border_checkpoint,burial_site,car_dealership,construction_site,crop_field,dam,\
                debris_or_rubble,educational_institution,electric_substation,factory_or_powerplant,fire_station,\
                flooded_road,fountain,gas_station,golf_course,ground_transportation_station,helipad,hospital,\
                impoverished_settlement,interchange,lake_or_pond,lighthouse,military_facility,\
                multi-unit_residential,nuclear_powerplant,office_building,oil_or_gas_facility,park,\
                parking_lot_or_garage,place_of_worship,police_station,port,prison,race_track,railway_bridge,\
                recreational_facility,road_bridge,runway,shipyard,shopping_mall,single-unit_residential,\
                smokestack,solar_farm,space_facility,stadium,storage_tank,surface_mine,swimming_pool,toll_booth,tower,\
                tunnel_opening,waste_disposal,water_treatment_facility,wind_farm,zoo").replace(' ', '').split(',')
        else:
            category_list = cat_list
            if bg_flag:
                category_list.append('false_detection')

        low_weight_list = ("wind_farm,tunnel_opening,solar_farm,nuclear_powerplant,military_facility,crop_field,airport,\
            flooded_road,debris_or_rubble,single-unit_residential").replace(' ', '').split(',')

        high_weight_list = ("border_checkpoint,construction_site,educational_institution,factory_or_powerplant,fire_station,\
            police_station,gas_station,smokestack,tower,road_bridge").replace(' ', '').split(',')

        category_set = set(category_list)
        low_weight_set = set(low_weight_list)
        high_weight_set = set(high_weight_list)

        weight_cat_dict = {
            0.6: low_weight_list,
            1.4: high_weight_list,
            1.0: category_set.difference(low_weight_set.union(high_weight_set)) 
        }
        if bg_flag:
            weight_cat_dict[0.0] = ['false_detection']
            category_list.append('false_detection')

        cat_weight_dict = dict(sum([list(zip(v, len(v)*[k])) for k,v in weight_cat_dict.items()], []))

        cat_ind_dict = dict(zip(category_list, range(len(category_list))))
        ind_cat_dict = dict([(v,k) for k,v in cat_ind_dict.items()])
        self.LABEL_MAP = cat_ind_dict
        self.IDX_MAP = ind_cat_dict

        self.WEIGHT_TSR = torch.FloatTensor(len(ind_cat_dict))
        for ind, cat in ind_cat_dict.items():
            self.WEIGHT_TSR[ind] = cat_weight_dict[cat]

        self.NUM_CLASSES = len(category_list)

        self.CONST_FACTOR = 1e6
