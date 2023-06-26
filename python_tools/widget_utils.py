
from IPython.display import YouTubeVideo
from IPython.display import display
import ipywidgets as widgets
"""
How to install the widget for ipynb

!pip3 install ipywidgets
!jupyter nbextension enable --py widgetsnbextension

#import ipywidgets as widgets
#from IPython.display import display



Ex: 
pixel_layout = 0
item_layout = widgets.Layout(margin=f'{pixel_layout}px {pixel_layout}px {pixel_layout}px {pixel_layout}px')
dropdown_year = widgets.ToggleButtons(
    description = "Dropdown",
    options = ["yes","no","yes partial"]
)

toggle = widgets.RadioButtons(
    options = ["option1","option2"]
)

input_widgets = widgets.VBox([
    dropdown_year,toggle,toggle,toggle,
    
],layout = item_layout,)

notes_box = widgets.Textarea(
    placehold = "Type some notes here",
    description = "Notes",
    layout = Layout(
        width = "100%",
        #min_height = "300px"
    )
)
btn = widgets.Button(description='Save Validation to Dict')
def btn_eventhandler(obj):
    print('Hello from the {} button!'.format(obj.description))
btn.on_click(btn_eventhandler)
total_widgets = widgets.VBox([input_widgets,notes_box,btn])
display(total_widgets)


Categories of Buttons

"""
text_numbers_widgets = [
    "BoundedIntText",
    "BoundedFloatText",
    "IntText",
    "FloatText",
]

boolean_widgets = [
    "ToggleButton",
    "CheckBox",
    "Valid",
    "Select",
    "SelectionSlider",
    "SelectionRangeSlider",
    "ToggleButtons", #multiple buttons but can only select one
    "SelectMutliple",
]

"""
For options you can set the  string label
or put label value pairs

Ex: 
widgets.Dropdown(
    options=[('One', 1), ('Two', 2), ('Three', 3)],
    value=2,
    description='Number:',
)
"""
selection_widgets = [
    "Dropdown",
    "RadioButtons"
]

"""


"""
string_widgets = [
    "Text",
    "Textarea",
    "Combobox",
    "Password",
    "Label"
    
]

action_item_widgets = [
    "Button",
]

animation_widgets = [
    "Play"
]

date_time_widgets = [
    "DatePicker",
    "TimePicker",
    "DateTimePicker"
    
]

color_picker_widgets = [
    "ColorPicker"
]

"""
Notes: 
Each container has a children attributes that can be set
"""
container_widgets = [
    "Box",
    "HBox",
    "VBox",
    "GridBox",
]

multi_container_widgets = [
    "Tabs", #can show different panels
    "Accordian", #have the arrows that can control if displays or not
]

#import ipywidgets as widgets
#from IPython.display import display

def set_label(w,label):
    w.description = label

def example_label_widget():
    """
    Use label when want to create a custom label
    to use in a widget
    """
    return widgets.HBox([widgets.Label(value="The $m$ in $E=mc^2$:"), widgets.FloatSlider()])


def example_image(filepath):
    file = open(f"{filepath}", "rb")
    image = file.read()
    widgets.Image(
        value=image,
        format='png',
        width=300,
        height=400,
    )
    
def example_animation():
    play = widgets.Play(
    value = 50,
    min = 0,
    max = 100,
    step = 1,
    interval = 50,
    description="Press Play,"
    )

    slider = widgets.FloatSlider()
    widgets.jslink((play,'value'),(slider,'value'))
    widgets.HBox([play,slider])
    
def example_color_picker():
    j = widgets.ColorPicker(
        concise=False,
        description='Pick a color',
        value='blue',
        disabled=False
    )
    
def example_tabs():
    tab_contents = ['P0', 'P1', 'P2', 'P3', 'P4']
    children = [widgets.Text(description=name) for name in tab_contents]
    tab = widgets.Tab()
    tab.children = children
    tab.titles = [str(i) for i in range(len(children))]
    return tab

def example_accordian():
    accordion = widgets.Accordion(
    children=[widgets.IntSlider(), widgets.Text()],
    titles=('Slider', 'Text'))
    return accordion

#from IPython.display import YouTubeVideo
#from IPython.display import display
def example_display_youtube_video(
    link = 'eWzY2nGfkXk'
    ):
    
    out = widgets.Output(layout={'border': '1px solid black'})
    display(out)
    with out:
        display(YouTubeVideo(link))
        
def example_function_to_output_widget(
    w = None,
    clear_output = True, #whether to clear output every time write to it
    wait = True, #only clears output once new output sent
    ):
    
    if w is None:
        w = widgets.Output()
        display(w)

    @w.capture(clear_output = True,wait = True)
    def function_with_captured_output(
        n=10):
        for i in range(n):
            print(i, 'Hello world!')

        
def clear_output(w,wait = False):
    """
    Wait will wait until the next time something
    is sent to it
    """
    w.clear_output(wait = wait)
    
def example_linking_interactive_function_with_output():
    """
    Pseudocode: 
    1) Create slider
    2) Creates function to output string of sliders
    3) Creates output widget
    4) Stack sliders and output widget in one widget
    """
    a = widgets.IntSlider(description="a")
    b = widgets.IntSlider(description='b')
    c = widgets.IntSlider(description='c')

    def f(a,b,c):
        print(f"{a}*{b}*{c}={a*b*c}")

    out = widgets.interactive_output(f,dict(a=a,b=b,c=c))
    widgets.HBox([widgets.VBox([a,b,c]),out])
    
def example_add_function_to_button():
    """
    Psuedocode: 
    1) Create button and output
    2) Display Button and Output
    3) Create a function that brings to the output
    4) Adds the function to button widget on .on_click funct
    """
    button = widgets.Button(description="Click Me!")
    output = widgets.Output()

    display(button, output)

    @output.capture(clear_output=True,wait=True)
    def on_button_clicked(b):
        print("Button clicked.")

    button.on_click(on_button_clicked)
    
def example_add_function_to_value_change():
    """
    Purpose: The function that runs on the change 
    should have an argument that accepts a dictionary
    holding the information of the change


    
    """
    int_range = widgets.IntSlider()
    output = widgets.Output()

    display(int_range, output)
    
    @output.capture(clear_output=True,wait = True)
    def on_value_change(change):
        print(change['new'])
        
    int_range.observe(on_value_change,names='value')
    
def add_on_change_func(
    w,
    f,
    out = None,
    #names="value",
    #type="change",#type of notification to filter by
    ):
    if out is not None:
        #out.capture(clear_output=True,wait = True)
        def new_func(*args,**kwargs):
            with out:
                return f(*args,**kwargs)
        curr_func = new_func
    else:
        curr_func = f
    
    w.observe(
        curr_func,
        #names=names,
        #type=type
    )
    
# ---------- Linking attributes ------
"""
Psueodocde: 
1) Create the widgets
2) Use .link or .dlink (only the downstream is linked to upstream) to connect certain attributes of widgets
3) Display


Notes:
1) using the jslink and jsdlink allow attributes to be linked
even when 
"""

def example_link_both_directions():
    sliders1, slider2 = widgets.IntSlider(description='Slider 1'),\
                    widgets.IntSlider(description='Slider 2')
    l = widgets.link((sliders1, 'value'), (slider2, 'value'))
    display( sliders1, slider2)
    
def example_link_directional():
    sliders1, slider2 = widgets.IntSlider(description='Slider 1'),\
                    widgets.IntSlider(description='Slider 2')
    l = widgets.dlink((sliders1, 'value'), (slider2, 'value'))
    display( sliders1, slider2)
    
def unlink(link):
    link.unlink()




