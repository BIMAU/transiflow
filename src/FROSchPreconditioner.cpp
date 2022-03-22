#ifndef FVM_FROSCH_PRECONDITIONER_CPP
#define FVM_FROSCH_PRECONDITIONER_CPP

namespace FROSch {

    IfpackPreconditioner::IfpackPreconditioner(Teuchos::RCP<const Epetra_RowMatrix> matrix,
                                               Teuchos::RCP<Teuchos::ParameterList> &parameterList):
        Matrix_ (matrix),
        ParameterList_ (parameterList)
    {
        using namespace Teuchos;
        using namespace Xpetra;
        using Xpetra::Matrix;

        // Convert the matrix
        RCP<const Epetra_CrsMatrix> crsMatrix = rcp_dynamic_cast<const Epetra_CrsMatrix>(matrix);
        RCP<Epetra_CrsMatrix> crsMatrix_nonconst = rcp_const_cast<Epetra_CrsMatrix>(crsMatrix);
        RCP<CrsMatrix<double,int,FROSch::DefaultGlobalOrdinal,KokkosClassic::DefaultNode::DefaultNodeType> > xCrsMatrix = rcp(new EpetraCrsMatrixT<FROSch::DefaultGlobalOrdinal,KokkosClassic::DefaultNode::DefaultNodeType>(crsMatrix_nonconst));
        // this is a wrapper to turn the object into an Xpetra object
        RCP<const Matrix<double,int,FROSch::DefaultGlobalOrdinal,KokkosClassic::DefaultNode::DefaultNodeType> > xMatrix = rcp(new CrsMatrixWrap<double,int,FROSch::DefaultGlobalOrdinal,KokkosClassic::DefaultNode::DefaultNodeType>(xCrsMatrix));
        // this is an Xpetra::Matrix that allows 'viewing' the matrix like a block matrix, for instance

        FROSchPreconditioner_.reset(new OneLevelPreconditioner<double,int>(xMatrix,ParameterList_));
        //ConstXMapPtr repeatedMap = extractRepeatedMap(comm,underlyingLib);
    }

    int IfpackPreconditioner::Initialize()
    {
        FROSchPreconditioner_->initialize(ParameterList_->get("Overlap",1));
        IsInitialized_ = true;
        return 0;
    }

    int IfpackPreconditioner::Compute()
    {
        FROSCH_ASSERT(IsInitialized_==true,"IsInitialized_==false.");
        FROSchPreconditioner_->compute();
        IsComputed_ = true;
        return 0;
    }

    int IfpackPreconditioner::ApplyInverse(const Epetra_MultiVector &X,
                                           Epetra_MultiVector &Y) const
    {
        using namespace Teuchos;
        using namespace Xpetra;

        FROSCH_ASSERT(IsComputed_==true,"IsInitialized_==false.");

        RCP<const MultiVector<double,int,FROSch::DefaultGlobalOrdinal> > xX = rcp(new EpetraMultiVectorT<FROSch::DefaultGlobalOrdinal,KokkosClassic::DefaultNode::DefaultNodeType>(rcpFromRef(const_cast<Epetra_MultiVector&>(X))));
        RCP<MultiVector<double,int,FROSch::DefaultGlobalOrdinal> > xY = rcp(new EpetraMultiVectorT<FROSch::DefaultGlobalOrdinal, KokkosClassic::DefaultNode::DefaultNodeType>(rcpFromRef(Y)));

        FROSchPreconditioner_->apply(*xX,*xY);
        return 0;
    }

} // namespace FROSch

#endif
