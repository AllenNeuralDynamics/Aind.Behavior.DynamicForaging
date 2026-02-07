json-schema
-------------
The following json-schemas are used as the format definition of the input for this task. They are the result of the `Pydantic`` models defined in `src/aind_behavior_dynamic_foraging`, and are also used to generate `src/Extensions/AindBehaviorDynamicForaging.cs` via `Bonsai.Sgen`.

`Download Schema <https://raw.githubusercontent.com/AllenNeuralDynamics/Aind.Behavior.DynamicForaging/main/src/DataSchemas/aind_behavior_dynamic_foraging.json>`_

Task Logic Schema
~~~~~~~~~~~~~~~~~
.. jsonschema:: https://raw.githubusercontent.com/AllenNeuralDynamics/Aind.Behavior.DynamicForaging/main/src/DataSchemas/aind_behavior_dynamic_foraging.json#/$defs/AindDynamicForagingTaskLogic
   :lift_definitions:
   :auto_reference:


Rig Schema
~~~~~~~~~~~~~~
.. jsonschema:: https://raw.githubusercontent.com/AllenNeuralDynamics/Aind.Behavior.DynamicForaging/main/src/DataSchemas/aind_behavior_dynamic_foraging.json#/$defs/AindDynamicForagingRig
   :lift_definitions:
   :auto_reference:
