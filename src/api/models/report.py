from pydantic import BaseModel

class FaultReport(BaseModel):
    fault_type:          str        # Center, Donut, Edge-Loc etc or "unknown"
    affected_parameter:  str        # which sensor triggered — gas_flow, temp, pressure
    lead_time_ms:        float      # from your LSTM lead time result
    severity:            str        # low, medium, high
    confidence:          float      # 0.0 - 1.0
    recommended_action:  str        # what the engineer should do
    urgency:             str        # immediate, scheduled, monitor
    mode:                int        # 1 or 2
    
