# -*- coding: utf-8 -*-

class Extractor():
    
    @staticmethod
    def extract_x(p_fields,source_location):
        (src_y, src_x) = source_location
        p = list()
        for frame in p_fields:
            p.append(frame[src_y,:])
        return p
        