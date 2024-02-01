from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput


class CalculatorApp(App):

    def build(self):
        self.equation = ''
        self.result = TextInput(text='', readonly=True)
        buttons = [
            ['7', '8', '9', '/'],
            ['4', '5', '6', '*'],
            ['1', '2', '3', '-'],
            ['.', '0', '=', '+']
        ]
        layout = BoxLayout(orientation='vertical')
        layout.add_widget(self.result)

        for row in buttons:
            button_row = BoxLayout()
            for label in row:
                button = Button(text=label)
                button.bind(on_press=self.on_button_press)
                button_row.add_widget(button)
            layout.add_widget(button_row)

        return layout

    def on_button_press(self, instance):
        if instance.text == '=':
            try:
                self.result.text = str(eval(self.equation))
            except Exception as e:
                self.result.text = 'Error'
            self.equation = ''
        else:
            self.equation += instance.text


if __name__ == '__main__':
    CalculatorApp().run()