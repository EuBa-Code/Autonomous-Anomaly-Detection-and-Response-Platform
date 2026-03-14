from manim import *

class DataFlowML(Scene):
    def construct(self):
        # Creazione dei componenti dell'architettura
        input_layer = Square(color=BLUE).shift(LEFT * 4).scale(0.7)
        model = Rectangle(color=WHITE, height=2, width=3).set_fill(GREY, opacity=0.5)
        output_layer = Circle(color=GREEN).shift(RIGHT * 4).scale(0.7)
        
        # Etichette
        input_text = Text("Input Data").next_to(input_layer, UP)
        model_text = Text("ML Model").move_to(model.get_center())
        output_text = Text("Prediction").next_to(output_layer, UP)
        
        # Frecce di collegamento
        arrow1 = Arrow(input_layer.get_right(), model.get_left(), buff=0.1)
        arrow2 = Arrow(model.get_right(), output_layer.get_left(), buff=0.1)
        
        self.add(input_layer, model, output_layer, input_text, model_text, output_text, arrow1, arrow2)

        # Animazione del "Dato" (un punto che scorre)
        data_packet = Dot(color=YELLOW).move_to(input_layer.get_center())
        
        self.play(FadeIn(data_packet))
        # Il dato si muove verso il modello
        self.play(data_packet.animate.move_to(model.get_center()), run_time=1.5)
        self.play(Indicate(model)) # Il modello elabora
        # Il dato esce verso l'output
        self.play(data_packet.animate.move_to(output_layer.get_center()), run_time=1.5)
        self.play(FadeOut(data_packet))
        self.wait()

