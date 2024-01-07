import { NgModule } from '@angular/core';
import { CommonModule, TitleCasePipe } from '@angular/common';
import { RouterModule } from '@angular/router';


import { NgbModule } from '@ng-bootstrap/ng-bootstrap';

import * as PlotlyJS from 'plotly.js-dist-min';
import { PlotlyModule } from 'angular-plotly.js';

PlotlyModule.plotlyjs = PlotlyJS;

@NgModule({
    imports: [ RouterModule, CommonModule, NgbModule, PlotlyModule ],
    declarations: [   ],
    exports: [ PlotlyModule ],
    providers: [TitleCasePipe]
})

export class SharedModule {}
