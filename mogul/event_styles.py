EVENT_STYLES={
    'course_out':('レーン逸脱','#ff5252'),
    'course_exit':('コース逸脱','#c62828'),
    'stumble_candidate':('躓き候補','#ffa726'),
    'fall_candidate':('転倒候補','#ec407a'),
    'balance_candidate':('姿勢異常候補','#ffa726'),
}

def event_style(kind):
    return EVENT_STYLES.get(kind,('異常候補','#ffa726'))
