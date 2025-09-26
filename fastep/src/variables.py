STATE_VARIABLES = ['month', 'hour', 'outdoor_temperature', 'outdoor_humidity', 'wind_speed', 'wind_direction',
                   'direct_solar_radiation', 'htg_setpoint_living', 'htg_setpoint_kitchen', 'htg_setpoint_bed1', 'htg_setpoint_bed2', 'htg_setpoint_bed3', 'air_temperature_living', 'air_temperature_kitchen', 'air_temperature_bed1', 'air_temperature_bed2', 'air_temperature_bed3', 'air_humidity_living', 'air_humidity_kitchen', 'air_humidity_bed1', 'air_humidity_bed2',
                   'air_humidity_bed3', 'heat_source_electricity_rate']

ACTION_VARIABLES = ['flow_rate_living', 'flow_rate_kitchen',
                    'flow_rate_bed1', 'flow_rate_bed2', 'flow_rate_bed3']

TARGET_VARIABLES = ['air_temperature_living', 'air_temperature_kitchen', 'air_temperature_bed1',
                    'air_temperature_bed2', 'air_temperature_bed3', 'heat_source_electricity_rate']
