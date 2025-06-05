def unfreeze_model_layers(model, layers_to_unfreeze):
    def apply_freeze_flags(module, name_prefix=""):
        for name, child in module.named_children():
            full_name = f"{name_prefix}.{name}" if name_prefix else name
            if full_name in layers_to_unfreeze:
                for param in child.parameters():
                    param.requires_grad = True
                print(f"Unfrozen layer: {full_name}")
            else:
                for param in child.parameters():
                    param.requires_grad = False
                apply_freeze_flags(child, full_name)

    apply_freeze_flags(model)

    print("\nTrainable parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f" - {name}")
