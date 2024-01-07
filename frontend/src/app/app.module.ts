import { CUSTOM_ELEMENTS_SCHEMA, NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';

import { AppComponent } from './app.component';
import { HttpClientModule } from '@angular/common/http';

import { CommonModule } from '@angular/common';
import { ExplorerDashboardComponent } from './modules/pages/dashboard/explorer.dashboard.component';
import { BrowserAnimationsModule } from '@angular/platform-browser/animations';
import { AgGridModule } from "@ag-grid-community/angular";
import { ModuleRegistry } from '@ag-grid-community/core';
import { ClientSideRowModelModule } from '@ag-grid-community/client-side-row-model';
import { ToastrModule } from 'ngx-toastr';
import { SharedModule } from './modules/shared/shared.module';
import { AppRoutes } from './app.routing';
import { RouterModule } from '@angular/router';
import { ReactiveFormsModule } from '@angular/forms';
import { PorscheDesignSystemModule } from '@porsche-design-system/components-angular';

ModuleRegistry.registerModules([ClientSideRowModelModule]);
@NgModule({
  declarations: [
    AppComponent,
    ExplorerDashboardComponent,
  ],
  imports: [
    CommonModule,
    BrowserModule,
    BrowserAnimationsModule,
    ReactiveFormsModule,
    RouterModule.forRoot(AppRoutes, {
      useHash: true
    }),
    HttpClientModule,
    AgGridModule,
    ToastrModule.forRoot(),
    SharedModule,
    PorscheDesignSystemModule
  ],
  providers: [],
  bootstrap: [AppComponent],
  schemas: [CUSTOM_ELEMENTS_SCHEMA]
})
export class AppModule { }
